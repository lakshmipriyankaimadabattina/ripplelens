import json, glob, statistics as st, collections, sys
import numpy as np
from scipy.stats import wilcoxon
R = [json.loads(l) for f in sorted(glob.glob("out/*_m3_results.jsonl")) for l in open(f)]
repos = ["commons-io","commons-collections","commons-lang","commons-compress","pdfbox"]
M = ["random","same_package","static","static_plus","co_change","combined","abl_no_hist","abl_no_struct","abl_no_name"]
K = ["P@5","R@5","F1@5","R@10","Hit@10","MRR","Effort"]
def agg(rows):
    o = {}
    for m in M:
        o[m] = {k: (st.median if k=="Effort" else st.mean)([r["metrics"][m][k] for r in rows]) for k in K}
    return o
out = {"n_instances": len(R), "n_commits": len({r["sha"] for r in R})}
out["pooled"] = agg(R); out["per_repo"] = {}
for rp in repos:
    rr = [r for r in R if r["repo"]==rp]
    out["per_repo"][rp] = {"n_instances": len(rr), "n_commits": len({r["sha"] for r in rr}), "metrics": agg(rr)}
# commit-level means for tests
bycommit = collections.defaultdict(list)
for r in R: bycommit[(r["repo"], r["sha"])].append(r)
def cm(m, k="MRR"): return [st.mean(x["metrics"][m][k] for x in v) for v in bycommit.values()]
pairs = [("co_change","same_package"),("static_plus","static"),("co_change","static_plus"),("combined","co_change"),("combined","static_plus"),("static","same_package")]
ps = []
for a,b in pairs:
    x,y = cm(a), cm(b); p = wilcoxon(x,y).pvalue; ps.append((a,b,p,float(np.mean(x)-np.mean(y)), float(np.mean(np.array(x)>np.array(y))), float(np.mean(np.array(x)<np.array(y)))))
order = sorted(range(len(ps)), key=lambda i: ps[i][2]); holm = [None]*len(ps); prev=0
for rank,i in enumerate(order):
    holm[i] = min(1.0, max(prev, ps[i][2]*(len(ps)-rank))); prev = holm[i]
out["wilcoxon_MRR"] = [{"a":a,"b":b,"p":p,"p_holm":h,"mean_diff":d,"frac_a_better":fa,"frac_b_better":fb} for (a,b,p,d,fa,fb),h in zip(ps,holm)]
# by change type
bt = {}
for t in ["bugfix","refactor","feature","doc","other"]:
    rr = [r for r in R if r["ctype"]==t]
    if rr: bt[t] = {"n_commits": len({r["sha"] for r in rr}), "n_instances": len(rr), "metrics": {m: {k: st.mean(r["metrics"][m][k] for r in rr) for k in ("R@10","MRR")} for m in ("static_plus","co_change","combined","same_package")}}
out["by_type"] = bt
# by time third within each repo (by date order of commits)
thirds = collections.defaultdict(list)
for rp in repos:
    cs = sorted({(r["date"], r["sha"]) for r in R if r["repo"]==rp})
    n = len(cs); idx = {sha: min(2, int(3*i/n)) for i,(d,sha) in enumerate(cs)}
    for r in R:
        if r["repo"]==rp: thirds[idx[r["sha"]]].append(r)
out["by_third"] = {str(k): {"n_instances": len(v), "metrics": {m: {kk: st.mean(r["metrics"][m][kk] for r in v) for kk in ("R@10","MRR")} for m in ("static_plus","co_change","combined")}} for k,v in sorted(thirds.items())}
# seed variance (MRR sd across seeds within commit)
sd = {}
for m in ("static_plus","co_change","combined"):
    vals = [np.std([x["metrics"][m]["MRR"] for x in v]) for v in bycommit.values() if len(v)>1]
    sd[m] = float(np.mean(vals))
out["seed_sd_MRR"] = sd; out["commits_with_multi_seed"] = sum(1 for v in bycommit.values() if len(v)>1)
# observations
T = sum(len(r["truth"]) for r in R)
fl = lambda k: [x for r in R for x in r[k]]
d = fl("truth_dist")
out["obs"] = {"T": T, "unreach": 100*sum(1 for x in d if x is None)/T, "d1": 100*sum(1 for x in d if x==1)/T,
  "zero_co": 100*sum(1 for x in fl("truth_cochange_hist") if x==0)/T, "samepkg": 100*sum(fl("truth_same_package"))/T,
  "testfiles": 100*sum(fl("truth_is_test"))/T,
  "size1_commits": 100*sum(1 for v in bycommit.values() if len(v[0]["truth"])==1)/len(bycommit)}
u = [r for r in R if "union10" in r]
out["union"] = {"n": len(u), "median_size": st.median(len(r["union10"]) for r in u), "recall": sum(sum(r["truth_in_union"]) for r in u)/sum(len(r["truth"]) for r in u),
  "co10_recall": sum(sum(1 for t in r["truth"] if t in r["top20"]["co_change"][:10]) for r in u)/sum(len(r["truth"]) for r in u),
  "combined10_recall": sum(sum(1 for t in r["truth"] if t in r["top20"]["combined"][:10]) for r in u)/sum(len(r["truth"]) for r in u)}
# post-cutoff vs earlier (heuristic methods)
for flag in (True, False):
    rr = [r for r in R if r["post_cutoff"]==flag]
    out["post_cutoff" if flag else "pre_cutoff"] = {"n_instances": len(rr), "n_commits": len({r["sha"] for r in rr}), "metrics": agg(rr)}
out["ctype_counts"] = dict(collections.Counter(next(iter(v))["ctype"] for v in bycommit.values()))
json.dump(out, open("out/analysis.json","w"), indent=1)
print(json.dumps({k: out[k] for k in ("n_instances","n_commits","commits_with_multi_seed","seed_sd_MRR","obs","union","ctype_counts")}, indent=1))
print("POOLED"); 
for m in M: print(m, {k: round(v,3) for k,v in out["pooled"][m].items()})
for rp in repos:
    print(rp, out["per_repo"][rp]["n_commits"], out["per_repo"][rp]["n_instances"], {m: (round(out["per_repo"][rp]["metrics"][m]["MRR"],3), round(out["per_repo"][rp]["metrics"][m]["R@10"],3)) for m in ("same_package","static_plus","co_change","combined")})
print("WIL"); [print(w) for w in out["wilcoxon_MRR"]]
print("TYPE"); [print(t, v["n_commits"], {m: tuple(round(x,3) for x in vv.values()) for m,vv in v["metrics"].items()}) for t,v in out["by_type"].items()]
print("THIRD"); [print(t, v["n_instances"], {m: tuple(round(x,3) for x in vv.values()) for m,vv in v["metrics"].items()}) for t,v in out["by_third"].items()]
for f in ("pre_cutoff","post_cutoff"): print(f, out[f]["n_commits"], {m: (round(out[f]["metrics"][m]["MRR"],3), round(out[f]["metrics"][m]["R@10"],3)) for m in ("same_package","static_plus","co_change","combined")})
