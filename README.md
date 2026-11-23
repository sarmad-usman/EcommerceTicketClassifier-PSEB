# AI-Powered Ticket Classification Assistant

**Student:** Muhammad Sarmad Usman  
**ID:** TRN-2026-000296  
**Module:** 05 – AI Engineering Bootcamp  
**Scenario:** E-commerce customer support

## Project overview
This application accepts a support ticket, predicts one of exactly ten categories, and returns a short factual summary. Streamlit is the frontend, FastAPI is the API boundary, and `service.py` contains reusable classification and provider logic. The default mode is local deterministic mock mode so the complete project runs without credentials.

## Ten categories
See `data/category_definitions.md` for definitions and examples:
1. Billing and Payment
2. Order Status
3. Delivery Issues
4. Returns and Refunds
5. Account Access
6. Technical Issues
7. Product Information
8. Cancellation Requests
9. Damaged or Missing Items
10. Other Inquiries

## Dataset
`data/tickets.csv` contains 100 fictional, manually authored tickets: 80 development/reference rows and 20 fixed evaluation rows (two per category). The `split` column records the partition. No real customer information is used.

## Concepts from Module 05
- Prompt engineering: `prompts/prompt_v1.txt` and `prompts/prompt_v2.txt`; v2 adds grounding and prompt-injection boundaries.
- Structured output: `TicketResult` uses Pydantic with a fixed Literal category list and non-empty summary.
- Baselines: keyword rules and TF-IDF/cosine similarity.
- Evaluation: fixed held-out cases, accuracy, confusion matrices, summary review and error analysis.
- Application flow: Streamlit → FastAPI `/analyze` → reusable service → validated JSON response.
- Failure handling: blank/invalid input returns 422; unusable provider/service results are mapped to 502.
- Mock/live: mock is deterministic and offline. Optional live mode uses an OpenAI-compatible chat endpoint configured only on the backend.

## Important academic note
The default mock mode is a deterministic teaching fixture, not a trained model and not a live LLM. TF-IDF is a similarity baseline, not an LLM. Evaluation outputs are labeled accordingly. The rubric explains that supervised classifier training is optional and that mock scores must not be described as real LLM performance.

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the application
Open **Terminal 1**:
```powershell
uvicorn api:app --reload
```
Check `http://127.0.0.1:8000/health` and open `http://127.0.0.1:8000/docs`.

Open **Terminal 2**:
```powershell
streamlit run streamlit_app.py
```
The browser opens the Streamlit interface. The default API address is `http://127.0.0.1:8000`. If the API is hosted elsewhere, set `API_URL` before starting Streamlit.

## Run evaluation and tests
```powershell
python evaluation/run_evaluation.py
pytest -q
```
Evaluation files are saved in `evaluation/`: `metrics.csv`, three confusion matrices, `evaluation_results.csv`, `prompt_v2_errors.csv`, and `EVALUATION_REPORT.md`.

## Optional live model mode
The project runs fully in mock mode by default. To use an approved OpenAI-compatible endpoint, install the listed dependencies and set these variables in the **backend terminal only**:
```powershell
$env:LLM_MODE="live"
$env:OPENAI_API_KEY="YOUR_KEY"
$env:LLM_MODEL="YOUR_DEPLOYMENT_NAME"
$env:LLM_BASE_URL="YOUR_APPROVED_BASE_URL"
uvicorn api:app --reload
```
Never place a real key in source code, `.env` committed to GitHub, screenshots, or the ZIP. The key remains on the backend. Live mode is optional and requires a valid deployment and endpoint. The UI does not receive the key.

## Demo checklist
1. Submit a valid fictional ticket, e.g. “I was charged twice for the same order.”
2. Submit a blank ticket; Streamlit should ask for input and the API independently rejects blank text with 422.
3. Stop FastAPI and submit a ticket; Streamlit should show a backend connection error.
4. Demonstrate prompt-injection case `REG-03`; the prompt says customer ticket text is untrusted data.
5. Record whether the demo used mock or live mode and save screenshots of success and failure.

## Limitations
- Synthetic dataset is small and may not represent real support language.
- TF-IDF depends on word overlap and is not semantic understanding.
- Mock mode is not a generative LLM; summaries are extractive.
- Live model output may vary and requires schema validation and human review.
- Classification is a suggestion only. It must not issue refunds, cancel orders, or change accounts.

## Submission contents
Dataset, category definitions, source code, prompts, automated tests, fixed evaluation results, confusion matrices and this README are included.
