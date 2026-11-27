#!/usr/bin/env python3
"""RippleLens A2 baseline pipeline (file-level change impact prediction).

Usage: python ripplelens_baseline.py <repo_dir> <out_dir> [--max-test 300] [--seed 42]

Methods: random, same-package, static (import-graph distance), co-change (confidence).
Temporal protocol: statistics only from commits strictly older than the test commit;
test commits are drawn from the latest 30% of non-merge commits.
"""
import argparse, json, os, random, re, statistics, subprocess, sys, time
from collections import defaultdict

MAX_HIST_FILES = 30            # history commits larger than this are ignored for co-change stats
MIN_T, MAX_T = 2, 15           # test commit size filter (Java files)
SPLIT = 0.70                   # oldest 70% = history-only window, latest 30% = test window
KS = (5, 10)
BULK_RE = re.compile(r"format|whitespace|reformat|indent|trailing|tabs?\b|license header|copyright|spotless|checkstyle|(organi[sz]e|clean ?up|sort|optimi[sz]e) imports?|imports? (cleanup|order)", re.I)
PKG_RE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.M)
IMP_RE = re.compile(r"^\s*import\s+(static\s+)?([\w.]+?)(\.\*)?\s*;", re.M)


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, check=True).stdout.decode("utf-8", "ignore")


def read_history(repo):
    fmt = "COMMIT\t%H\t%ct\t%s"
    out = git(repo, "log", "--no-merges", "--reverse", "-M", "--name-status", f"--format={fmt}")
    commits, cur = [], None
    for line in out.splitlines():
        if line.startswith("COMMIT\t"):
            _, sha, ts, subj = line.split("\t", 3)
            cur = {"sha": sha, "ts": int(ts), "subject": subj, "changes": []}
            commits.append(cur)
        elif line.strip() and cur is not None:
            p = line.split("\t")
            st = p[0][0]
            if st == "R" or st == "C":
                old, new = p[1], p[2]
                if old.endswith(".java") or new.endswith(".java"):
                    cur["changes"].append(("R" if st == "R" else "A", old, new))
            elif p[1].endswith(".java"):
                cur["changes"].append((st, p[1], p[1]))
    return commits


class Blobs:
    def __init__(self, repo):
        self.repo = repo
        self.p = subprocess.Popen(["git", "-C", repo, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.cache = {}

    def parse(self, blob):
        if blob in self.cache:
            return self.cache[blob]
        self.p.stdin.write((blob + "\n").encode()); self.p.stdin.flush()
        hdr = self.p.stdout.readline().split()
        size = int(hdr[2]); data = self.p.stdout.read(size + 1)[:-1].decode("utf-8", "ignore")
        m = PKG_RE.search(data)
        pkg = m.group(1) if m else ""
        imps = [(bool(a), b, bool(c)) for a, b, c in IMP_RE.findall(data)]
        self.cache[blob] = (pkg, imps)
        return self.cache[blob]


def snapshot(repo, commit, blobs, pid):
    """Return {id: (path, pkg, imports)} for the Java files in the parent tree of `commit`."""
    out = git(repo, "ls-tree", "-r", f"{commit}^")
    files = {}
    unmapped = 0
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        if not path.endswith(".java"):
            continue
        blob = meta.split()[2]
        fid = pid.get(path)
        if fid is None:
            unmapped += 1
            continue
        pkg, imps = blobs.parse(blob)
        files[fid] = (path, pkg, imps)
    return files, unmapped


def build_graph(files):
    fq, bypkg = {}, defaultdict(list)
    for fid, (path, pkg, _) in files.items():
        stem = os.path.basename(path)[:-5]
        fq[(pkg + "." + stem) if pkg else stem] = fid
        bypkg[pkg].append(fid)
    adj = defaultdict(set)
    for fid, (_, _, imps) in files.items():
        for is_static, name, wild in imps:
            if wild:
                if name in fq:
                    tgts = [fq[name]]
                else:
                    tgts = bypkg.get(name, [])
            else:
                tgts, n = [], name
                while n:
                    if n in fq:
                        tgts = [fq[n]]; break
                    n = n.rpartition(".")[0]
            for t in tgts:
                if t != fid:
                    adj[fid].add(t); adj[t].add(fid)
    return adj


def bfs(adj, seed):
    dist, frontier, d = {seed: 0}, [seed], 0
    while frontier:
        d += 1; nxt = []
        for u in frontier:
            for v in adj.get(u, ()):
                if v not in dist:
                    dist[v] = d; nxt.append(v)
        frontier = nxt
    return dist


def metrics(ranked, truth):
    res, hits = {}, [1 if r in truth else 0 for r in ranked]
    T = len(truth)
    for k in KS:
        h = sum(hits[:k]); p = h / k; r = h / T
        res[f"P@{k}"], res[f"R@{k}"] = p, r
        res[f"F1@{k}"] = (2 * p * r / (p + r)) if (p + r) else 0.0
    res["MRR"] = next((1 / (i + 1) for i, x in enumerate(hits) if x), 0.0)
    last = max(i for i, x in enumerate(hits) if x) + 1 if sum(hits) == T else len(ranked)
    res["Effort"] = last
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo"); ap.add_argument("out")
    ap.add_argument("--max-test", type=int, default=300); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    name = os.path.basename(a.repo.rstrip("/"))
    commits = read_history(a.repo)
    N = len(commits); boundary = int(N * SPLIT)

    # ---- eligibility funnel (test window only) ----
    funnel = {"non_merge_commits": N, "test_window_commits": N - boundary}
    elig = []
    n_touch = n_size = n_rename = n_bulk = 0
    for i in range(boundary, N):
        c = commits[i]; nj = len(c["changes"])
        if nj == 0: continue
        n_touch += 1
        if not (MIN_T <= nj <= MAX_T): continue
        n_size += 1
        if sum(1 for s, *_ in c["changes"] if s == "R") * 2 > nj: n_rename += 1; continue
        if BULK_RE.search(c["subject"]): n_bulk += 1; continue
        elig.append(i)
    funnel.update({"touch_java": n_touch, "size_2_15_java": n_size, "dropped_rename_dominated": n_rename,
                   "dropped_bulk_subject": n_bulk, "eligible": len(elig)})
    rng = random.Random(a.seed)
    sample = set(rng.sample(elig, min(a.max_test, len(elig))))
    funnel["sampled_test_commits"] = len(sample)

    # ---- chronological replay ----
    blobs = Blobs(a.repo)
    pid, next_id = {}, 0
    cnt = defaultdict(int); co = defaultdict(lambda: defaultdict(int))
    results, unmapped_total, skipped = [], 0, defaultdict(int)
    for i, c in enumerate(commits):
        if i in sample:
            files, um = snapshot(a.repo, c["sha"], blobs, pid); unmapped_total += um
            changed_exist, changed_new = [], []
            for st, old, new in c["changes"]:
                if st == "A": changed_new.append(new)
                elif old in pid: changed_exist.append(pid[old])
            changed_exist = [x for x in dict.fromkeys(changed_exist) if x in files]
            if len(changed_exist) < 2:
                skipped["fewer_than_2_pre_existing_files"] += 1
            else:
                r = random.Random(f"{a.seed}-{c['sha']}")
                seed = r.choice(sorted(changed_exist))
                truth = set(changed_exist) - {seed}
                cand = [f for f in files if f != seed]
                adj = build_graph(files); dist = bfs(adj, seed)
                spkg = files[seed][1]
                def shuffled(xs):
                    xs = list(xs); r.shuffle(xs); return xs
                rank = {}
                rank["random"] = shuffled(cand)
                same = [f for f in cand if files[f][1] == spkg]
                rank["same_package"] = shuffled(same) + shuffled(f for f in cand if files[f][1] != spkg)
                reach = sorted((f for f in cand if f in dist), key=lambda f: dist[f])
                # random tie-break inside equal distance
                reach = [f for _, g in __import__("itertools").groupby(reach, key=lambda f: dist[f]) for f in shuffled(g)]
                rank["static"] = reach + shuffled(f for f in cand if f not in dist)
                nb = co.get(seed, {}); cs = cnt.get(seed, 0)
                withco = sorted((f for f in cand if nb.get(f, 0) > 0), key=lambda f: (-nb[f] / cs, -nb[f]))
                withco = [f for _, g in __import__("itertools").groupby(withco, key=lambda f: (nb[f] / cs, nb[f])) for f in shuffled(g)]
                rank["co_change"] = withco + shuffled(f for f in cand if nb.get(f, 0) == 0)
                m = {k: metrics(v, truth) for k, v in rank.items()}
                inv = {v: k for k, v in pid.items()}
                tl = list(truth)
                results.append({
                    "repo": name, "sha": c["sha"], "subject": c["subject"], "n_candidates": len(cand),
                    "seed": files[seed][0], "seed_has_history": cs > 0,
                    "truth": [files[t][0] for t in tl], "n_new_files_in_commit": len(changed_new),
                    "truth_in_import_graph": [t in dist for t in tl],
                    "truth_dist": [dist.get(t) for t in tl],
                    "truth_cochange_hist": [nb.get(t, 0) for t in tl],
                    "truth_same_package": [files[t][1] == spkg for t in tl],
                    "truth_is_test": [("src/test" in files[t][0]) for t in tl],
                    "seed_is_test": "src/test" in files[seed][0],
                    "top10": {k: [files[f][0] for f in v[:10]] for k, v in rank.items()},
                    "metrics": m,
                })
        # ---- apply commit to history state ----
        ids = []
        for st, old, new in c["changes"]:
            if st == "A":
                pid[new] = next_id; next_id += 1; ids.append(pid[new])
            elif st == "R":
                fid = pid.pop(old, None)
                if fid is None: fid = next_id; next_id += 1
                pid[new] = fid; ids.append(fid)
            elif st == "D":
                fid = pid.pop(old, None)
                if fid is not None: ids.append(fid)
            else:
                fid = pid.get(old)
                if fid is None: fid = next_id; next_id += 1; pid[old] = fid
                ids.append(fid)
        ids = list(dict.fromkeys(ids))
        if len(ids) <= MAX_HIST_FILES:
            for x in ids:
                cnt[x] += 1
            for x in ids:
                for y in ids:
                    if x != y: co[x][y] += 1

    # ---- aggregate ----
    methods = ["random", "same_package", "static", "co_change"]
    keys = [f"{m}@{k}" for m in ("P", "R", "F1") for k in KS] if False else [f"{m}@{k}" for k in KS for m in ("P", "R", "F1")] + ["MRR", "Effort"]
    agg = {m: {} for m in methods}
    for m in methods:
        for k in keys:
            vals = [r["metrics"][m][k] for r in results]
            agg[m][k] = statistics.mean(vals) if k != "Effort" else statistics.median(vals)
    flat = lambda key: [x for r in results for x in r[key]]
    tg = flat("truth_in_import_graph"); tc = flat("truth_cochange_hist"); sp = flat("truth_same_package"); ist = flat("truth_is_test")
    obs = {
        "n_eval_commits": len(results),
        "n_true_files": len(tg),
        "pct_true_files_reachable_in_import_graph": 100 * sum(tg) / len(tg),
        "pct_true_files_with_prior_cochange_with_seed": 100 * sum(1 for x in tc if x > 0) / len(tc),
        "pct_true_files_same_package_as_seed": 100 * sum(sp) / len(sp),
        "pct_true_files_that_are_test_files": 100 * sum(ist) / len(ist),
        "pct_seeds_without_history": 100 * sum(1 for r in results if not r["seed_has_history"]) / len(results),
        "pct_commits_with_new_files": 100 * sum(1 for r in results if r["n_new_files_in_commit"] > 0) / len(results),
        "median_candidates": statistics.median(r["n_candidates"] for r in results),
        "median_true_set_size": statistics.median(len(r["truth"]) for r in results),
        "median_import_distance_when_reachable": statistics.median([d for d in flat("truth_dist") if d is not None]),
    }
    # Wilcoxon (paired, per-commit MRR) co-change vs each other method
    try:
        from scipy.stats import wilcoxon
        wil = {}
        for o in ("random", "same_package", "static"):
            x = [r["metrics"]["co_change"]["MRR"] for r in results]; y = [r["metrics"][o]["MRR"] for r in results]
            wil[f"co_change_vs_{o}"] = wilcoxon(x, y).pvalue
        obs["wilcoxon_MRR_p"] = wil
    except Exception as e:
        obs["wilcoxon_error"] = str(e)
    head = git(a.repo, "rev-parse", "HEAD").strip()
    summary = {"repo": name, "head_sha": head, "head_date": git(a.repo, "log", "-1", "--format=%cs").strip(),
               "total_commits": int(git(a.repo, "rev-list", "--count", "HEAD")), "java_files_at_head": len([l for l in git(a.repo, "ls-tree", "-r", "HEAD", "--name-only").splitlines() if l.endswith(".java")]),
               "split_boundary_sha": commits[boundary]["sha"], "split_boundary_date": time.strftime("%Y-%m-%d", time.gmtime(commits[boundary]["ts"])),
               "funnel": funnel, "skipped_after_sampling": dict(skipped), "unmapped_paths": unmapped_total,
               "metrics": agg, "observations": obs, "runtime_s": round(time.time() - t0, 1)}
    json.dump(summary, open(os.path.join(a.out, f"{name}_summary.json"), "w"), indent=1)
    with open(os.path.join(a.out, f"{name}_results.jsonl"), "w") as f:
        for r in results: f.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
