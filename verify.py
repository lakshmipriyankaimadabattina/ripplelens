#!/usr/bin/env python3
"""Compare a regenerated analysis.json with the committed one (default tolerance 1e-9)."""
import json, sys
new, ref, tol = sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 1e-9
A, B = json.load(open(new)), json.load(open(ref)); bad = 0
def walk(a, b, path=""):
    global bad
    if isinstance(b, dict):
        for k in b: walk(a.get(k) if isinstance(a, dict) else None, b[k], f"{path}/{k}")
    elif isinstance(b, list):
        for i, (x, y) in enumerate(zip(a or [], b)): walk(x, y, f"{path}[{i}]")
    elif isinstance(b, (int, float)) and not isinstance(b, bool):
        if a is None or abs(a - b) > tol: bad += 1; print("MISMATCH", path, a, b)
walk(A, B)
print("VERIFY OK: all values match" if not bad else f"VERIFY FAILED: {bad} mismatches"); sys.exit(1 if bad else 0)
