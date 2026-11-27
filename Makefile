# make reproduce : clone pinned repos, rerun pipeline + analysis, compare with committed results
PY=python3
repos:
	bash scripts/clone_repos.sh
run: repos
	mkdir -p out
	for r in commons-io commons-collections commons-lang commons-compress pdfbox; do $(PY) src/ripplelens_m3.py repos/$$r out > out/$$r.log & done; wait
analyze:
	$(PY) src/analyze_m3.py
reproduce: run analyze
	$(PY) verify.py out/analysis.json results/m3/analysis.json
check-analysis:
	mkdir -p out && cp results/m3/*_m3_results.jsonl out/ && $(PY) src/analyze_m3.py && $(PY) verify.py out/analysis.json results/m3/analysis.json
view-test:
	cd view && npm install jsdom && node test_view.js
