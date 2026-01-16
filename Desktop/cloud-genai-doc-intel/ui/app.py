import json
import time
import os
from datetime import datetime

import boto3
import streamlit as st

st.set_page_config(page_title="Doc Intelligence (AWS + GenAI)", layout="wide")

st.title("📄 Cloud-Based GenAI Document Intelligence System")
st.caption("Upload PDFs → S3 triggers Lambda → outputs saved in processed/")

# ---------- CONFIG ----------
# Option A: hardcode bucket here (quick)
#BUCKET = st.secrets.get("BUCKET_NAME", "") if hasattr(st, "secrets") else ""
BUCKET = ""

# You can also paste manually in UI if you prefer

RAW_PREFIX = "raw/"
PROCESSED_PREFIX = "processed/"

# ---------- AWS CLIENT ----------
@st.cache_resource
def get_s3_client():
    # Uses your AWS credentials (AWS CLI config or env vars)
    return boto3.client("s3")

s3 = get_s3_client()

def list_objects(prefix: str):
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    keys = []
    for p in pages:
        for obj in p.get("Contents", []):
            keys.append((obj["Key"], obj["LastModified"], obj["Size"]))
    # newest first
    keys.sort(key=lambda x: x[1], reverse=True)
    return keys

def get_text(key: str) -> str:
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    return obj["Body"].read().decode("utf-8", errors="replace")

def key_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket_name, Key=key)
        return True
    except Exception:
        return False

def upload_pdf(file_bytes: bytes, filename: str) -> str:
    safe_name = filename.replace(" ", "_")
    s3_key = f"{RAW_PREFIX}{int(time.time())}_{safe_name}"
    s3.put_object(Bucket=bucket_name, Key=s3_key, Body=file_bytes)
    return s3_key

def base_from_raw_key(raw_key: str) -> str:
    # raw/<timestamp>_name.pdf -> processed/<timestamp>_name.*
    name = raw_key.split("/")[-1]
    base = name.rsplit(".", 1)[0]
    return base

# ---------- UI ----------
bucket_name = st.text_input("S3 Bucket Name", value=BUCKET, placeholder="doc-intel-manvi-2026")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("⬆️ Upload a PDF")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    upload_btn = st.button("Upload to S3 (raw/)", type="primary", disabled=(uploaded is None or not bucket_name))

    if upload_btn and uploaded is not None:
        raw_key = upload_pdf(uploaded.getvalue(), uploaded.name)
        st.success(f"Uploaded to s3://{bucket_name}/{raw_key}")
        st.session_state["last_raw_key"] = raw_key

with col2:
    st.subheader("🧾 Check processing output")
    last_raw_key = st.session_state.get("last_raw_key", "")
    raw_key_input = st.text_input(
        "Raw S3 key (auto-filled after upload, or paste one)",
        value=last_raw_key,
        placeholder="raw/1700000000_myfile.pdf"
    )

    if raw_key_input:
        base = base_from_raw_key(raw_key_input)
        txt_key = f"{PROCESSED_PREFIX}{base}.txt"
        meta_key = f"{PROCESSED_PREFIX}{base}.json"
        structured_key = f"{PROCESSED_PREFIX}{base}.structured.json"
        err_key = f"{PROCESSED_PREFIX}{base}.structured.error.json"

        st.write("Expected outputs:")
        st.code("\n".join([txt_key, meta_key, structured_key, err_key]))

        poll = st.button("🔄 Refresh status", disabled=not bucket_name)

        # Always check once when key is present, and also when Refresh is clicked
        if poll or True:
            status_cols = st.columns(4)

            status_cols[0].metric(".txt", "✅" if key_exists(txt_key) else "⏳")
            status_cols[1].metric(".json", "✅" if key_exists(meta_key) else "⏳")
            status_cols[2].metric(".structured.json", "✅" if key_exists(structured_key) else "⏳")
            status_cols[3].metric(".structured.error.json", "⚠️" if key_exists(err_key) else "—")

            if key_exists(structured_key):
                st.success("Structured output is ready!")
            elif key_exists(err_key):
                st.warning("LLM step failed — error file exists.")
            else:
                st.info("Still processing… click Refresh again in a few seconds.")

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("📁 Recent raw uploads")
    if bucket_name:
        try:
            raw_objs = list_objects(RAW_PREFIX)[:20]
            for k, lm, sz in raw_objs:
                st.write(f"• `{k}`  —  {lm.strftime('%Y-%m-%d %H:%M:%S')}  —  {sz} bytes")
        except Exception as e:
            st.error(f"Could not list raw/: {e}")
    else:
        st.info("Enter bucket name to browse raw/")

with right:
    st.subheader("✅ Recent processed outputs")
    if bucket_name:
        try:
            proc_objs = list_objects(PROCESSED_PREFIX)[:30]
            for k, lm, sz in proc_objs:
                st.write(f"• `{k}`  —  {lm.strftime('%Y-%m-%d %H:%M:%S')}  —  {sz} bytes")
        except Exception as e:
            st.error(f"Could not list processed/: {e}")
    else:
        st.info("Enter bucket name to browse processed/")

st.divider()

st.subheader("🔎 View an output file")

view_key = st.text_input("Paste an S3 key from processed/ to preview", placeholder="processed/....structured.json")

if view_key and bucket_name:
    try:
        content = get_text(view_key)
        if view_key.endswith(".json"):
            try:
                st.json(json.loads(content))
            except Exception:
                st.code(content)
        else:
            st.text_area("File preview", content, height=400)
    except Exception as e:
        st.error(f"Could not open {view_key}: {e}")

st.caption("Tip: Make sure your AWS credentials are configured locally (AWS CLI) so boto3 can access S3.")
