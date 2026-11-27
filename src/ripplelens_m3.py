#!/usr/bin/env python3
"""RippleLens M3 pipeline (file-level change impact prediction).

Usage: python ripplelens_m3.py <repo_dir> <out_dir> [--max-test 300] [--seed 42] [--seeds-per-commit 3]

Methods: random, same_package, static (import distance), static_plus (distance + same-package + low-degree
tie-break), co_change (confidence), combined (gradient boosting over structural, historical and naming features;
trained ONLY on commits from the history window that precedes the test window) and three feature-group ablations.
Temporal protocol: all statistics for a commit come from commits strictly older than it.
"""
import argparse, itertools, json, os, random, re, statistics, subprocess, time
from collections import defaultdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

MAX_HIST_FILES, MIN_T, MAX_T, SPLIT = 30, 2, 15, 0.70
CUTOFF_TS = 1782864000  # 2026-07-01T00:00:00Z (after the LLM's knowledge cutoff, end of June 2026)
BULK_RE = re.compile(r"format|whitespace|reformat|indent|trailing|tabs?\b|license header|copyright|spotless|checkstyle|(organi[sz]e|clean ?up|sort|optimi[sz]e) imports?|imports? (cleanup|order)", re.I)
PKG_RE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.M)
IMP_RE = re.compile(r"^\s*import\s+(static\s+)?([\w.]+?)(\.\*)?\s*;", re.M)
METHODS = ["random", "same_package", "static", "static_plus", "co_change", "combined",
           "abl_no_hist", "abl_no_struct", "abl_no_name"]
KS = (5, 10)
FN = ["d", "deg_c", "deg_s", "same_pkg", "same_dir", "same_module", "conf", "sup", "conf_rev", "jacc", "log_cnt_c",
      "test_pair", "name_overlap", "c_is_test", "s_is_test"]
GROUPS = {"hist": ["conf", "sup", "conf_rev", "jacc", "log_cnt_c"],
          "struct": ["d", "deg_c", "deg_s", "same_pkg", "same_dir", "same_module"],
          "name": ["test_pair", "name_overlap", "c_is_test", "s_is_test"]}


def ctype(s):
    s = s.lower()
    if re.search(r"\b(javadoc|typo|typos|comments?|documentation|docs?|readme|site)\b", s): return "doc"
    if re.search(r"\b(fix|fixes|fixed|fixing|bug|npe|nullpointer|regression|crash|incorrect|wrong|broken|leak|overflow|fail|fails|failure|failing)\b", s): return "bugfix"
    if re.search(r"(refactor|rename|\bmove[sd]?\b|extract|inline|simplif|clean ?up|\bremove|unused|deprecat|replace|reorgani|consolidat|migrat|\bupgrade|\buse\b|\bavoid)", s): return "refactor"
    if re.search(r"\b(add|adds|added|new|support|introduce|implement|feature|enhance|allow|allows|provide|improve|extend)\b", s): return "feature"
    return "other"


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, check=True).stdout.decode("utf-8", "ignore")


def read_history(repo):
    out = git(repo, "log", "--no-merges", "--reverse", "-M", "--name-status", "--format=COMMIT\t%H\t%ct\t%s")
    commits, cur = [], None
    for line in out.splitlines():
        if line.startswith("COMMIT\t"):
            _, sha, ts, subj = line.split("\t", 3)
            cur = {"sha": sha, "ts": int(ts), "subject": subj, "changes": []}; commits.append(cur)
        elif line.strip() and cur is not None:
            p = line.split("\t"); st = p[0][0]
            if st in "RC":
                if p[1].endswith(".java") or p[2].endswith(".java"):
                    cur["changes"].append(("R" if st == "R" else "A", p[1], p[2]))
            elif p[1].endswith(".java"):
                cur["changes"].append((st, p[1], p[1]))
    return commits


class Blobs:
    def __init__(self, repo):
        self.p = subprocess.Popen(["git", "-C", repo, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.cache = {}

    def parse(self, blob):
        if blob in self.cache: return self.cache[blob]
        self.p.stdin.write((blob + "\n").encode()); self.p.stdin.flush()
        size = int(self.p.stdout.readline().split()[2]); data = self.p.stdout.read(size + 1)[:-1].decode("utf-8", "ignore")
        m = PKG_RE.search(data)
        self.cache[blob] = (m.group(1) if m else "", [(bool(a), b, bool(c)) for a, b, c in IMP_RE.findall(data)])
        return self.cache[blob]


def snapshot(repo, commit, blobs, pid):
    files, um = {}, 0
    for line in git(repo, "ls-tree", "-r", f"{commit}^").splitlines():
        meta, path = line.split("\t", 1)
        if not path.endswith(".java"): continue
        fid = pid.get(path)
        if fid is None: um += 1; continue
        pkg, imps = blobs.parse(meta.split()[2]); files[fid] = (path, pkg, imps)
    return files, um


def build_graph(files):
    fq, bypkg = {}, defaultdict(list)
    for fid, (path, pkg, _) in files.items():
        fq[(pkg + "." + os.path.basename(path)[:-5]) if pkg else os.path.basename(path)[:-5]] = fid
        bypkg[pkg].append(fid)
    adj = defaultdict(set)
    for fid, (_, _, imps) in files.items():
        for _, name, wild in imps:
            if wild: tgts = [fq[name]] if name in fq else bypkg.get(name, [])
            else:
                tgts, n = [], name
                while n:
                    if n in fq: tgts = [fq[n]]; break
                    n = n.rpartition(".")[0]
            for t in tgts:
                if t != fid: adj[fid].add(t); adj[t].add(fid)
    return adj


def bfs(adj, seed):
    dist, fr, d = {seed: 0}, [seed], 0
    while fr:
        d += 1; nx = []
        for u in fr:
            for v in adj.get(u, ()):
                if v not in dist: dist[v] = d; nx.append(v)
        fr = nx
    return dist


def metrics(ranked, truth):
    hits = [1 if r in truth else 0 for r in ranked]; T = len(truth); res = {}
    for k in KS:
        h = sum(hits[:k]); p = h / k; r = h / T
        res[f"P@{k}"], res[f"R@{k}"] = p, r
        res[f"F1@{k}"] = (2 * p * r / (p + r)) if (p + r) else 0.0
    res["Hit@10"] = 1.0 if sum(hits[:10]) else 0.0
    res["MRR"] = next((1 / (i + 1) for i, x in enumerate(hits) if x), 0.0)
    res["Effort"] = (max(i for i, x in enumerate(hits) if x) + 1) if sum(hits) == T else len(ranked)
    return res


STEM_RE = re.compile(r"(Tests?|TestCase|ITCase|IT)$")
TOK_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
_tok = {}
def tokens(path):
    if path not in _tok: _tok[path] = {t.lower() for t in TOK_RE.findall(os.path.basename(path)[:-5])} - {"test", "tests", "it", "case"}
    return _tok[path]
def module(path): return path.split("/src/")[0] if "/src/" in path else path.split("/")[0]


def feature_matrix(seed, cands, files, adj, dist, cnt, co):
    sp, spkg = files[seed][0], files[seed][1]
    s_stem = STEM_RE.sub("", os.path.basename(sp)[:-5]); s_test = "src/test" in sp or "/test/" in sp
    nb = co.get(seed, {}); cs = cnt.get(seed, 0); sdir, smod, stok = os.path.dirname(sp), module(sp), tokens(sp)
    deg_s = len(adj.get(seed, ())); X = np.zeros((len(cands), len(FN)), dtype=np.float32)
    for r, c in enumerate(cands):
        cp = files[c][0]; n = nb.get(c, 0); cc = cnt.get(c, 0); c_test = "src/test" in cp or "/test/" in cp
        ctok = tokens(cp); u = len(stok | ctok)
        X[r] = (min(dist.get(c, 9), 9), len(adj.get(c, ())), deg_s, files[c][1] == spkg, os.path.dirname(cp) == sdir,
                module(cp) == smod, n / cs if cs else 0.0, n, n / cc if cc else 0.0, n / (cs + cc - n) if (cs + cc - n) > 0 else 0.0,
                np.log1p(cc), STEM_RE.sub("", os.path.basename(cp)[:-5]) == s_stem and c_test != s_test,
                len(stok & ctok) / u if u else 0.0, c_test, s_test)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo"); ap.add_argument("out")
    ap.add_argument("--max-test", type=int, default=300); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds-per-commit", type=int, default=3); ap.add_argument("--max-train", type=int, default=400)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    name = os.path.basename(a.repo.rstrip("/")); commits = read_history(a.repo)
    N = len(commits); boundary = int(N * SPLIT)

    def passes(i):
        c = commits[i]; nj = len(c["changes"])
        if not (MIN_T <= nj <= MAX_T): return "size"
        if sum(1 for s, *_ in c["changes"] if s == "R") * 2 > nj: return "rename"
        if BULK_RE.search(c["subject"]): return "bulk"
        return "ok"

    funnel = {"non_merge_commits": N, "test_window_commits": N - boundary,
              "touch_java": sum(1 for i in range(boundary, N) if commits[i]["changes"])}
    cnts = defaultdict(int)
    for i in range(boundary, N):
        if commits[i]["changes"]: cnts[passes(i)] += 1
    funnel.update({"size_2_15_java": cnts["ok"] + cnts["rename"] + cnts["bulk"], "dropped_rename_dominated": cnts["rename"],
                   "dropped_bulk_subject": cnts["bulk"], "eligible": cnts["ok"]})
    elig = [i for i in range(boundary, N) if commits[i]["changes"] and passes(i) == "ok"]
    rng = random.Random(a.seed)
    sample = set(rng.sample(elig, min(a.max_test, len(elig))))
    extra = {i for i in elig if commits[i]["ts"] >= CUTOFF_TS and i not in sample}
    funnel.update({"sampled_test_commits": len(sample), "extra_post_cutoff_commits": len(extra)})
    sample |= extra
    lo = int(0.35 * N); pool = [i for i in range(lo, boundary) if passes(i) == "ok"]
    train_idx = set(random.Random(a.seed + 1).sample(pool, min(a.max_train, len(pool))))
    funnel["train_commits_sampled"] = len(train_idx)

    blobs = Blobs(a.repo); pid, nid = {}, 0
    cnt = defaultdict(int); co = defaultdict(lambda: defaultdict(int))
    Xtr, ytr, models = [], [], {}
    results, skipped, unmapped = [], defaultdict(int), 0
    cols = {g: [j for j, f in enumerate(FN) if f not in GROUPS[g]] for g in GROUPS}

    def fit():
        X = np.vstack(Xtr); y = np.concatenate(ytr)
        def mk(idx): return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=6, random_state=a.seed).fit(X[:, idx], y)
        models["combined"] = (mk(list(range(len(FN)))), list(range(len(FN))))
        for g in GROUPS: models[f"abl_no_{g}"] = (mk(cols[g]), cols[g])
        return X.shape[0], int(y.sum())

    train_rows = (0, 0)
    for i, c in enumerate(commits):
        if i == boundary and Xtr: train_rows = fit()
        if i in train_idx or i in sample:
            files, um = snapshot(a.repo, c["sha"], blobs, pid); unmapped += um
            cex = [pid[o] for st, o, n_ in c["changes"] if st != "A" and o in pid]
            cex = [x for x in dict.fromkeys(cex) if x in files]
            n_new = sum(1 for st, *_ in c["changes"] if st == "A")
            if len(cex) >= 2:
                r = random.Random(f"{a.seed}-{c['sha']}")
                adj = build_graph(files)
                if i in train_idx:
                    seed = r.choice(sorted(cex)); truth = set(cex) - {seed}; dist = bfs(adj, seed); nb = co.get(seed, {})
                    cand = [f for f in files if f != seed]
                    plaus = [f for f in cand if nb.get(f, 0) > 0 or dist.get(f, 99) <= 2 or files[f][1] == files[seed][1] or f in truth]
                    r.shuffle(plaus); sel = set(plaus[:200]) | truth
                    rest = [f for f in cand if f not in sel]; r.shuffle(rest); sel |= set(rest[:100])
                    sel = sorted(sel)
                    Xtr.append(feature_matrix(seed, sel, files, adj, dist, cnt, co)); ytr.append(np.array([f in truth for f in sel], dtype=np.int8))
                else:
                    seeds = r.sample(sorted(cex), min(a.seeds_per_commit, len(cex)))
                    for si, seed in enumerate(seeds):
                        truth = set(cex) - {seed}; cand = [f for f in files if f != seed]
                        dist = bfs(adj, seed); spkg = files[seed][1]; nb = co.get(seed, {}); cs = cnt.get(seed, 0)
                        def shuf(xs): xs = list(xs); r.shuffle(xs); return xs
                        def grp(xs, key): return [y for _, g in itertools.groupby(xs, key=key) for y in shuf(g)]
                        rank = {"random": shuf(cand)}
                        rank["same_package"] = shuf(f for f in cand if files[f][1] == spkg) + shuf(f for f in cand if files[f][1] != spkg)
                        reach = sorted((f for f in cand if f in dist), key=lambda f: dist[f])
                        rank["static"] = grp(reach, lambda f: dist[f]) + shuf(f for f in cand if f not in dist)
                        sk = lambda f: (dist.get(f, 99), files[f][1] != spkg, len(adj.get(f, ())))
                        rank["static_plus"] = grp(sorted(shuf(cand), key=sk), sk)
                        withco = sorted((f for f in cand if nb.get(f, 0) > 0), key=lambda f: (-nb[f] / cs, -nb[f]))
                        rank["co_change"] = grp(withco, lambda f: (nb[f] / cs, nb[f])) + shuf(f for f in cand if nb.get(f, 0) == 0)
                        order = shuf(cand); X = feature_matrix(seed, order, files, adj, dist, cnt, co)
                        for m in ("combined", "abl_no_hist", "abl_no_struct", "abl_no_name"):
                            if m in models:
                                mod, idx = models[m]; s = mod.predict_proba(X[:, idx])[:, 1]
                                rank[m] = [order[j] for j in np.argsort(-s, kind="stable")]
                        met = {k: metrics(v, truth) for k, v in rank.items()}
                        tl = sorted(truth); P = lambda f: files[f][0]
                        rec = {"repo": name, "sha": c["sha"], "date": time.strftime("%Y-%m-%d", time.gmtime(c["ts"])), "post_cutoff": c["ts"] >= CUTOFF_TS,
                               "extra": i in extra, "subject": c["subject"], "ctype": ctype(c["subject"]), "seed_idx": si, "n_seeds": len(seeds),
                               "n_candidates": len(cand), "seed": P(seed), "truth": [P(t) for t in tl], "n_new_files_in_commit": n_new,
                               "seed_has_history": cs > 0, "truth_dist": [dist.get(t) for t in tl], "truth_cochange_hist": [nb.get(t, 0) for t in tl],
                               "truth_same_package": [files[t][1] == spkg for t in tl], "truth_is_test": ["src/test" in P(t) or "/test/" in P(t) for t in tl],
                               "top20": {k: [P(f) for f in v[:20]] for k, v in rank.items()}, "metrics": met}
                        if si == 0 and "combined" in rank:
                            u = []
                            for m in ("co_change", "static_plus", "same_package", "combined"):
                                for f in rank[m][:10]:
                                    if P(f) not in u: u.append(P(f))
                            rec["union10"] = sorted(u); rec["truth_in_union"] = [t in set(u) for t in rec["truth"]]
                        results.append(rec)
            else:
                skipped["fewer_than_2_pre_existing_files"] += 1
        # ---- apply commit to history state ----
        ids = []
        for st, old, new in c["changes"]:
            if st == "A": pid[new] = nid; nid += 1; ids.append(pid[new])
            elif st == "R":
                fid = pid.pop(old, None)
                if fid is None: fid = nid; nid += 1
                pid[new] = fid; ids.append(fid)
            elif st == "D":
                fid = pid.pop(old, None)
                if fid is not None: ids.append(fid)
            else:
                fid = pid.get(old)
                if fid is None: fid = nid; nid += 1; pid[old] = fid
                ids.append(fid)
        ids = list(dict.fromkeys(ids))
        if len(ids) <= MAX_HIST_FILES:
            for x in ids: cnt[x] += 1
            for x in ids:
                for y in ids:
                    if x != y: co[x][y] += 1

    head = git(a.repo, "rev-parse", "HEAD").strip()
    summary = {"repo": name, "head_sha": head, "head_date": git(a.repo, "log", "-1", "--format=%cs").strip(),
               "total_commits": int(git(a.repo, "rev-list", "--count", "HEAD")),
               "java_files_at_head": sum(1 for l in git(a.repo, "ls-tree", "-r", "HEAD", "--name-only").splitlines() if l.endswith(".java")),
               "split_boundary_sha": commits[boundary]["sha"], "split_boundary_date": time.strftime("%Y-%m-%d", time.gmtime(commits[boundary]["ts"])),
               "funnel": funnel, "skipped_after_sampling": dict(skipped), "unmapped_paths": unmapped,
               "train_rows": train_rows[0], "train_positives": train_rows[1], "n_instances": len(results),
               "n_commits": len({r["sha"] for r in results}), "runtime_s": round(time.time() - t0, 1)}
    json.dump(summary, open(os.path.join(a.out, f"{name}_m3_summary.json"), "w"), indent=1)
    with open(os.path.join(a.out, f"{name}_m3_results.jsonl"), "w") as f:
        for r in results: f.write(json.dumps(r) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
