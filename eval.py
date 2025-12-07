import os
import json
import argparse
from datasets import Dataset, load_dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from docval import chat, verify

def run_ragas(questions_json: str):
    with open(questions_json, "r", encoding="utf-8") as f:
        qa_items = json.load(f)
    rows = []
    for item in qa_items:
        q = item["question"]
        gt = item.get("ground_truth", "")
        r = chat(q)
        rows.append({"question": q, "answer": r["answer"], "contexts": [r["context"]], "ground_truths": [gt]})
    ds = Dataset.from_list(rows)
    res = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
    print(json.dumps(res.to_dict(), indent=2))

def run_hallucination(questions_json: str):
    with open(questions_json, "r", encoding="utf-8") as f:
        qs = json.load(f)
    out = []
    supported = 0
    for q in qs:
        r = chat(q)
        v = verify(q, r["answer"], r["context"]) if r["context"] else {"verdict": "unsupported", "confidence": 0.0}
        out.append({"question": q, "answer": r["answer"], "verdict": v["verdict"], "confidence": v["confidence"], "docs": r["docs"]})
        if v["verdict"] == "supported":
            supported += 1
    idx = supported / max(1, len(out))
    print(json.dumps({"hallucination_index": 1 - idx, "results": out}, indent=2))

def run_truthful():
    ds = load_dataset("truthful_qa", "generation")
    from docval import get_chat_model
    llm = get_chat_model()
    results = []
    for row in ds["validation"][:50]:
        q = row["question"]
        ans = llm.invoke(q).content
        results.append({"question": q, "answer": ans})
    print(json.dumps({"results": results}, indent=2))

def main():
    parser = argparse.ArgumentParser(prog="eval")
    sub = parser.add_subparsers(dest="cmd")
    p_r = sub.add_parser("ragas")
    p_r.add_argument("questions_json")
    p_h = sub.add_parser("hallucination")
    p_h.add_argument("questions_json")
    sub.add_parser("truthful")
    args = parser.parse_args()
    if args.cmd == "ragas":
        run_ragas(args.questions_json)
    elif args.cmd == "hallucination":
        run_hallucination(args.questions_json)
    elif args.cmd == "truthful":
        run_truthful()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

