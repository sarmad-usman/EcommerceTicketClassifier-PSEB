# Evaluation report

## Evaluation protocol
Twenty fixed evaluation tickets (two per category) are held out from the 80 development/reference tickets. Labels were assigned when the synthetic dataset was created and are not changed after predictions.
All methods use the same held-out cases. Prompt v1 is represented by a deterministic keyword fixture; prompt v2 mock mode uses TF-IDF/cosine similarity against development tickets. These are teaching fixtures/baselines, not measured performance of a live LLM.

## Metrics
| method                 |   accuracy |   n_eval | result_type                                                                 |
|:-----------------------|-----------:|---------:|:----------------------------------------------------------------------------|
| Keyword baseline       |       0.75 |       20 | deterministic local fixture / similarity baseline, not live LLM performance |
| Prompt v1 mock fixture |       0.75 |       20 | deterministic local fixture / similarity baseline, not live LLM performance |
| Prompt v2 mock fixture |       0.5  |       20 | deterministic local fixture / similarity baseline, not live LLM performance |

## Confusion matrices
See the three CSV files in this folder. Rows are expected labels and columns are predicted labels.

## Error analysis
Prompt v2 mock fixture produced 10 errors out of 20 evaluation tickets. See `prompt_v2_errors.csv` for individual cases.
Typical limitations: (1) tickets mentioning multiple issues can be assigned to the nearest wording rather than the customer's true primary intent; (2) short or ambiguous tickets may have low similarity and still receive a category; (3) TF-IDF relies on lexical overlap and may miss paraphrases; (4) a keyword baseline can trigger on incidental words and cannot reliably understand context.

## Summary review
The local mock summary is extractive: it repeats a shortened version of the supplied ticket. This is intentionally conservative and avoids claiming an action was completed. Human review is still required.

## Regression cases
The five fixed regression cases are in `tests/regression_cases.json`. Re-run them after changing prompts, category definitions or provider settings.

## Interpretation
Do not present these deterministic results as real LLM accuracy. A live run must record the model/deployment identifier, mode, prompt version, date, and the actual observed outputs.