# Cloud-Based GenAI Document Intelligence System (AWS + Streamlit)

A serverless document intelligence pipeline: upload PDFs → AWS S3 triggers Lambda → text + metadata + GenAI structured JSON saved back to S3 → Streamlit UI displays status and previews outputs.

## What it does
1. **Upload PDF** from Streamlit UI (uploads to `s3://<bucket>/raw/`)
2. **S3 event** triggers AWS Lambda automatically
3. Lambda:
   - extracts text using **pypdf**
   - calls an **open-source LLM** via Hugging Face Router (OpenAI-compatible API)
   - writes outputs to `s3://<bucket>/processed/`

## Outputs (per document)
- `processed/<name>.txt` — extracted text
- `processed/<name>.json` — metadata (source, pages, char count)
- `processed/<name>.structured.json` — structured extraction (GenAI)
- `processed/<name>.structured.error.json` — error details if LLM step fails (with retries + JSON repair)

## Architecture
Streamlit UI → S3 (`raw/`) → Lambda (extract + LLM) → S3 (`processed/`) → UI previews results

## Tech Stack
- AWS S3 (storage + event triggers)
- AWS Lambda (serverless processing)
- CloudWatch Logs (observability)
- pypdf (PDF text extraction)
- Hugging Face Router (open-source LLM inference)
- Streamlit (UI)

## Setup (UI)
### Prerequisites
- Python 3.10+
- AWS credentials configured locally (IAM user keys)
- An S3 bucket with the Lambda trigger already set up

Do NOT commit credentials or tokens. Secrets are excluded via `.gitignore`.
