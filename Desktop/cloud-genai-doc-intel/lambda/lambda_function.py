import os
import json
import time
import random
import re
import urllib.parse
import urllib.request
import urllib.error
from io import BytesIO

import boto3
from pypdf import PdfReader

s3 = boto3.client("s3")


def extract_json_object(text: str) -> dict:
    """
    Extract the first top-level JSON object from model output and lightly repair common issues.
    """
    if not text:
        raise ValueError("Empty model output")

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in output: {text[:500]}")

    candidate = text[start:end + 1].strip()

    # Remove trailing commas before } or ]
    candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)

    # Replace smart quotes if any
    candidate = candidate.replace("“", "\"").replace("”", "\"").replace("’", "'")

    return json.loads(candidate)


def call_hf_extract(text: str) -> tuple[dict, str]:
    token = os.environ.get("HF_API_TOKEN")
    model = os.environ.get("HF_MODEL", "HuggingFaceTB/SmolLM3-3B:hf-inference")

    if not token:
        raise RuntimeError("HF_API_TOKEN is missing in environment variables")

    text = (text or "")[:8000]

    def hf_chat(messages):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 800,
        }

        req = urllib.request.Request(
            url="https://router.huggingface.co/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        return data["choices"][0]["message"]["content"].strip()

    # 1) First attempt: extract JSON
    base_messages = [
        {
            "role": "system",
            "content": "Return ONLY valid JSON. No markdown, no comments, no trailing commas, no extra text.",
        },
        {
            "role": "user",
            "content": f"""
Return ONLY valid JSON.

Schema:
{{
  "document_type": "resume|invoice|homework|research_paper|other",
  "title": null,
  "summary": "",
  "key_points": [],
  "entities": {{
    "people": [],
    "emails": [],
    "phone_numbers": [],
    "organizations": [],
    "dates": [],
    "amounts": [],
    "locations": []
  }},
  "fields": {{
    "resume": {{
      "candidate_name": null,
      "skills": [],
      "education": [],
      "experience": []
    }},
    "invoice": {{
      "invoice_number": null,
      "vendor": null,
      "invoice_date": null,
      "total_amount": null
    }}
  }}
}}

Document text:
{text}
""".strip(),
        },
    ]

    max_attempts = 3
    base_delay = 1.0
    last_err = None

    for attempt in range(1, max_attempts + 1):
        raw_output = ""
        try:
            raw_output = hf_chat(base_messages)
            structured = extract_json_object(raw_output)
            return structured, raw_output

        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                pass
            last_err = {"code": e.code, "reason": str(e.reason), "body": body}

            if e.code in (429, 500, 502, 503, 504):
                sleep_s = base_delay * (2 ** (attempt - 1)) + random.random()
                print(f"HF HTTP {e.code}. Retry {attempt}/{max_attempts} in {sleep_s:.2f}s")
                time.sleep(sleep_s)
                continue
            raise

        except Exception as e:
            # 2) If JSON parsing fails, do ONE "repair" pass
            try:
                repair_messages = [
                    {
                        "role": "system",
                        "content": "You are a JSON repair bot. Return ONLY valid JSON. No markdown, no extra text.",
                    },
                    {
                        "role": "user",
                        "content": f"Fix this to be valid JSON and return ONLY the JSON:\n\n{raw_output}",
                    },
                ]
                repaired = hf_chat(repair_messages)
                structured = extract_json_object(repaired)
                return structured, repaired
            except Exception as e2:
                last_err = {"error": str(e), "repair_error": str(e2)}
                sleep_s = base_delay * (2 ** (attempt - 1)) + random.random()
                print(f"HF parse/repair error. Retry {attempt}/{max_attempts} in {sleep_s:.2f}s: {last_err}")
                time.sleep(sleep_s)

    raise RuntimeError(f"HF call failed after retries: {last_err}")


def lambda_handler(event, context):
    record = event["Records"][0]
    bucket = record["s3"]["bucket"]["name"]
    key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])

    if not key.startswith("raw/") or not key.lower().endswith(".pdf"):
        print(f"Skipping non-target object: {key}")
        return {"statusCode": 200, "body": "skipped"}

    print(f"Processing PDF: s3://{bucket}/{key}")

    obj = s3.get_object(Bucket=bucket, Key=key)
    pdf_bytes = obj["Body"].read()

    reader = PdfReader(BytesIO(pdf_bytes))
    pages_text = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            pages_text.append(t)

    extracted_text = "\n\n".join(pages_text)

    base_name = key.split("/")[-1].rsplit(".", 1)[0]
    text_key = f"processed/{base_name}.txt"
    meta_key = f"processed/{base_name}.json"
    structured_key = f"processed/{base_name}.structured.json"
    err_key = f"processed/{base_name}.structured.error.json"

    # Save extracted text + metadata
    s3.put_object(Bucket=bucket, Key=text_key, Body=extracted_text.encode("utf-8"))
    s3.put_object(
        Bucket=bucket,
        Key=meta_key,
        Body=json.dumps(
            {
                "source_pdf": f"s3://{bucket}/{key}",
                "pages": len(reader.pages),
                "chars": len(extracted_text),
                "note": "Text extracted using pypdf (best for text-based PDFs). Scanned PDFs may be empty.",
            },
            indent=2,
        ).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"Saved extracted text to s3://{bucket}/{text_key}")

    if len(extracted_text.strip()) == 0:
        print("No extractable text found. Skipping LLM step.")
        return {"statusCode": 200, "body": "no_text"}

    # LLM step (graceful failure)
    raw_llm_output = None
    try:
        structured, raw_llm_output = call_hf_extract(extracted_text)

        s3.put_object(
            Bucket=bucket,
            Key=structured_key,
            Body=json.dumps(structured, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        print(f"Saved structured JSON to s3://{bucket}/{structured_key}")
        return {"statusCode": 200, "body": "processed_with_llm"}

    except Exception as e:
        err_payload = {
            "source_pdf": f"s3://{bucket}/{key}",
            "error": str(e),
            "hint": "If HF returns 429/503, it may be busy. Try later or use a smaller PDF.",
            "raw_llm_output_snippet": (raw_llm_output or "")[:2000],
        }
        s3.put_object(
            Bucket=bucket,
            Key=err_key,
            Body=json.dumps(err_payload, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        print(f"LLM step failed; saved error to s3://{bucket}/{err_key}")
        return {"statusCode": 200, "body": "processed_but_llm_failed"}
