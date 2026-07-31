"""Eval harness: runs questions.json through run_rag, scores keyword grounding + refusals.
Usage: cd backend && python eval/run_eval.py [--tag baseline]
Saves: eval/results_<tag>.json  Prints: per-question table + summary (pass rate, avg latencies).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv  # noqa: E402

from app.agents.orchestrator import run_rag  # noqa: E402

load_dotenv()

REFUSAL_MARKERS = ("don't have enough information", "doesn't look answerable")


def score(item, res):
    ans = res["answer"].lower()
    if not item["should_answer"]:
        passed = any(m in ans for m in REFUSAL_MARKERS) or res["route"] == "reject"
        return passed, "refused" if passed else "answered-should-refuse"
    hits = [k for k in item["must_contain"] if k.lower() in ans]
    need = item.get("min_hits", len(item["must_contain"]))
    passed = len(hits) >= need and res["route"] == "answer"
    return passed, f"keywords {len(hits)}/{len(item['must_contain'])}"


def main():
    tag = "baseline"
    if "--tag" in sys.argv:
        tag = sys.argv[sys.argv.index("--tag") + 1]
    # Pause between questions so free-tier quotas recover (Gemini allows only a
    # few requests per minute). Without this, Q4+ dies with 429s.
    pause = 5.0
    if "--pause" in sys.argv:
        pause = float(sys.argv[sys.argv.index("--pause") + 1])
    min_pass = 0.0
    if "--min-pass" in sys.argv:
        min_pass = float(sys.argv[sys.argv.index("--min-pass") + 1])
    with open("eval/questions.json") as f:
        questions = json.load(f)
    if "--limit" in sys.argv:
        # Smoke mode: grade only the first N questions (~3 LLM calls each).
        # A full 12-question run costs ~40+ API calls - save it for releases.
        questions = questions[:int(sys.argv[sys.argv.index("--limit") + 1])]
    rows = []
    t0 = time.time()
    for n, it in enumerate(questions):
        if n:
            time.sleep(pause)
        try:
            res = run_rag(it["q"])
            ok, detail = score(it, res)
            err = ""
        except Exception as e:  # noqa: BLE001 - eval must record, not crash
            res = {"route": "error", "grounding_score": 0, "timings_ms": {}}
            ok, detail, err = False, "exception", f"{type(e).__name__}: {str(e)[:200]}"
        rows.append({"id": it["id"], "q": it["q"], "pass": ok, "detail": detail,
                     "route": res.get("route"), "score": res.get("grounding_score"),
                     "retriever_model": res.get("retriever_model"),
                     "critic_model": res.get("critic_model"),
                     "timings_ms": res.get("timings_ms", {}), "error": err,
                     "answer_preview": res.get("answer", "")[:200]})
        t = res.get("timings_ms", {})
        print(f"[{'PASS' if ok else 'FAIL'}] {it['id']:6} route={res.get('route'):7} "
              f"score={res.get('grounding_score')} total={t.get('total')}ms ({detail}) {err}")
    passed = sum(1 for r in rows if r["pass"])
    totals = [r["timings_ms"].get("total", 0) for r in rows if r["timings_ms"].get("total")]
    def avg(k):
        vals = [r["timings_ms"].get(k, 0) for r in rows if r["timings_ms"]]
        return round(sum(vals) / max(len(rows), 1), 1)
    summary = {"tag": tag, "pass_rate": f"{passed}/{len(rows)}",
               "avg_ms": {"router": avg("router"), "retriever": avg("retriever"),
                          "embed": avg("embed"), "chat": avg("chat"),
                          "critic": avg("critic"),
                          "total": round(sum(totals) / max(len(totals), 1), 1) if totals else 0},
               "wall_s": round(time.time() - t0, 1)}
    print(json.dumps(summary, indent=2))
    with open(f"eval/results_{tag}.json", "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(f"saved eval/results_{tag}.json")
    # Quality gate: non-zero exit when pass rate drops below the floor,
    # so regressions fail CI instead of slipping through silently.
    rate = passed / max(len(rows), 1)
    if rate < min_pass:
        print(f"GATE FAILED: pass rate {rate:.0%} < required {min_pass:.0%}")
        sys.exit(1)
    print(f"gate ok: {rate:.0%} >= {min_pass:.0%}")


if __name__ == "__main__":
    main()
