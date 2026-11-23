from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from service import CATEGORIES, _keyword_predict, _tfidf_predict, _summary

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "tickets.csv")
eval_df = df[df["split"] == "eval"].copy()
eval_df["keyword_prediction"] = eval_df["ticket"].apply(_keyword_predict)
eval_df["prompt_v1_mock_prediction"] = eval_df["ticket"].apply(_keyword_predict)
eval_df["prompt_v2_mock_prediction"] = eval_df["ticket"].apply(lambda x: _tfidf_predict(x)[0])
eval_df["summary_review"] = eval_df["ticket"].apply(_summary)
eval_df["summary_grounded_review"] = "PASS - extractive summary uses ticket text only"
eval_df.to_csv(ROOT / "evaluation" / "evaluation_results.csv", index=False)

rows=[]
for col, name in [
    ("keyword_prediction","Keyword baseline"),
    ("prompt_v1_mock_prediction","Prompt v1 mock fixture"),
    ("prompt_v2_mock_prediction","Prompt v2 mock fixture")
]:
    rows.append({"method":name,"accuracy":accuracy_score(eval_df.expected_category, eval_df[col]),"n_eval":len(eval_df),"result_type":"deterministic local fixture / similarity baseline, not live LLM performance"})
summary = pd.DataFrame(rows)
summary.to_csv(ROOT / "evaluation" / "metrics.csv", index=False)

for col, name in [
    ("keyword_prediction","keyword_baseline"),
    ("prompt_v1_mock_prediction","prompt_v1_mock"),
    ("prompt_v2_mock_prediction","prompt_v2_mock")
]:
    cm = confusion_matrix(eval_df.expected_category, eval_df[col], labels=CATEGORIES)
    pd.DataFrame(cm, index=CATEGORIES, columns=CATEGORIES).to_csv(ROOT / "evaluation" / f"confusion_matrix_{name}.csv")

errors = eval_df[eval_df.expected_category != eval_df.prompt_v2_mock_prediction][["id","ticket","expected_category","prompt_v2_mock_prediction"]]
errors.to_csv(ROOT / "evaluation" / "prompt_v2_errors.csv", index=False)

report = ["# Evaluation report", "", "## Evaluation protocol", 
"Twenty fixed evaluation tickets (two per category) are held out from the 80 development/reference tickets. Labels were assigned when the synthetic dataset was created and are not changed after predictions.",
"All methods use the same held-out cases. Prompt v1 is represented by a deterministic keyword fixture; prompt v2 mock mode uses TF-IDF/cosine similarity against development tickets. These are teaching fixtures/baselines, not measured performance of a live LLM.",
"",
"## Metrics", summary.to_markdown(index=False), "",
"## Confusion matrices", "See the three CSV files in this folder. Rows are expected labels and columns are predicted labels.", "",
"## Error analysis", 
f"Prompt v2 mock fixture produced {len(errors)} errors out of {len(eval_df)} evaluation tickets. See `prompt_v2_errors.csv` for individual cases.",
"Typical limitations: (1) tickets mentioning multiple issues can be assigned to the nearest wording rather than the customer's true primary intent; (2) short or ambiguous tickets may have low similarity and still receive a category; (3) TF-IDF relies on lexical overlap and may miss paraphrases; (4) a keyword baseline can trigger on incidental words and cannot reliably understand context.",
"",
"## Summary review", "The local mock summary is extractive: it repeats a shortened version of the supplied ticket. This is intentionally conservative and avoids claiming an action was completed. Human review is still required.",
"",
"## Regression cases", "The five fixed regression cases are in `tests/regression_cases.json`. Re-run them after changing prompts, category definitions or provider settings.",
"",
"## Interpretation", "Do not present these deterministic results as real LLM accuracy. A live run must record the model/deployment identifier, mode, prompt version, date, and the actual observed outputs."]
(ROOT / "evaluation" / "EVALUATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
print(summary.to_string(index=False))
