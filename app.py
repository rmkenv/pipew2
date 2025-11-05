import os
import io
import time
import random
import socket
import logging
import requests
import httpx
import certifi
import re
from datetime import datetime, timedelta
from urllib.parse import urlparse
from typing import Optional, Dict, List

import streamlit as st

# Gemini SDK
from google import genai
from google.genai.errors import APIError, ClientError

# PDF extraction
try:
    import pypdf
    PDF_EXTRACTION_AVAILABLE = True
except ImportError:
    try:
        from PyPDF2 import PdfReader as PyPDF2Reader
        PDF_EXTRACTION_AVAILABLE = True
    except ImportError:
        PDF_EXTRACTION_AVAILABLE = False

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pipewrench-streamlit")

# --- Config/env ---
WHITELIST_URL = "https://raw.githubusercontent.com/rmkenv/pipewrench_mvp/main/custom_whitelist.json"
URL_REGEX = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

EMBEDDED_WHITELIST = [
    {"url": "https://www.epa.gov", "description": "EPA Regulations"},
    {"url": "https://www.osha.gov", "description": "OSHA Standards"},
    {"url": "https://www.fhwa.dot.gov", "description": "FHWA Standards"},
    {"url": "https://www.awwa.org", "description": "Water Standards"},
    {"url": "https://www.apwa.net", "description": "APWA Resources"},
    {"url": "https://www.asce.org", "description": "ASCE Standards"},
]

SESSION_EXPIRY_HOURS = int(os.getenv("SESSION_EXPIRY_HOURS", "24"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# --- Domain data (unchanged) ---
JOB_ROLES = {
    "general": {"name": "General DPW Staff", "context": "You are assisting general Department of Public Works staff with municipal infrastructure questions."},
    "director": {"name": "DPW Director", "context": "You are assisting a DPW Director with strategic planning, policy decisions, and departmental oversight."},
    "engineer": {"name": "Civil Engineer", "context": "You are assisting a licensed civil engineer with technical engineering standards, design specifications, and compliance requirements."},
    "project_manager": {"name": "Project Manager", "context": "You are assisting a project manager with construction management, scheduling, budgeting, and contractor coordination."},
    "inspector": {"name": "Construction Inspector", "context": "You are assisting a construction inspector with field inspection procedures, quality control, and compliance verification."},
    "maintenance": {"name": "Maintenance Supervisor", "context": "You are assisting a maintenance supervisor with asset management, preventive maintenance, and repair operations."},
    "environmental": {"name": "Environmental Compliance Officer", "context": "You are assisting an environmental compliance officer with EPA regulations, stormwater management, and environmental permits."},
    "safety": {"name": "Safety Officer", "context": "You are assisting a safety officer with OSHA compliance, workplace safety, and accident prevention."}
}

DEPARTMENT_PROMPTS = {
    "general_public_works": {"name": "General Public Works", "prompt": "You are a specialized AI assistant for Municipal Public Works departments. You help preserve institutional knowledge and provide accurate, cited information from approved sources."},
    "water_sewer": {"name": "Water & Sewer", "prompt": "You are a specialized AI assistant for Water & Sewer departments. You provide expert guidance on water distribution, wastewater treatment, and utility infrastructure."},
    "streets_highways": {"name": "Streets & Highways", "prompt": "You are a specialized AI assistant for Streets & Highways departments. You provide guidance on road maintenance, traffic management, and transportation infrastructure."},
    "environmental": {"name": "Environmental Compliance", "prompt": "You are a specialized AI assistant for Environmental Compliance. You help with EPA regulations, stormwater management, and environmental permitting."},
    "safety": {"name": "Safety & Training", "prompt": "You are a specialized AI assistant for Safety & Training. You provide guidance on OSHA compliance, workplace safety, and training programs."},
    "administration": {"name": "Administration & Planning", "prompt": "You are a specialized AI assistant for DPW Administration & Planning. You help with budgeting, project planning, and departmental management."}
}

# --- Whitelist helpers (mostly unchanged) ---
def fetch_whitelist() -> List[str]:
    try:
        resp = requests.get(WHITELIST_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return [e["url"] for e in data if "url" in e]
    except Exception as e:
        logger.warning(f"Failed to fetch external whitelist: {e}")
        return [e["url"] for e in EMBEDDED_WHITELIST]

def get_whitelisted_domains(whitelist_urls: List[str]):
    domains = set()
    for url in whitelist_urls:
        parsed = urlparse(url)
        if parsed.netloc:
            domains.add(parsed.netloc)
    return domains

def is_url_whitelisted(url: str, whitelist_urls: List[str]) -> bool:
    try:
        parsed = urlparse(url)
        for whitelisted_url in whitelist_urls:
            w = urlparse(whitelisted_url)
            if parsed.netloc == w.netloc and parsed.path.startswith(w.path):
                return True
    except Exception:
        return False
    return False

def enforce_whitelist_on_text(text: str, whitelist_urls: List[str]) -> str:
    if not text:
        return text
    bad_urls = []
    for url in set(URL_REGEX.findall(text)):
        url_clean = url.rstrip('.,);]')
        if not is_url_whitelisted(url_clean, whitelist_urls):
            bad_urls.append(url_clean)
    if not bad_urls:
        return text
    note = "\n\n[COMPLIANCE NOTICE]\n" \
           "The following URLs are not in the approved whitelist and must not be cited:\n" + \
           "\n".join(f"- {u}" for u in sorted(bad_urls)) + \
           "\n\nPlease revise citations to use only approved sources."
    return text + note

# --- PDF extraction (unchanged) ---
def extract_text_from_pdf(content: bytes) -> str:
    if not PDF_EXTRACTION_AVAILABLE:
        return "[ERROR: PDF extraction library not installed. Install pypdf or PyPDF2.]"
    try:
        try:
            import pypdf
            pdf = pypdf.PdfReader(io.BytesIO(content))
            text = ""
            for i, page in enumerate(pdf.pages):
                pt = page.extract_text()
                if pt:
                    text += f"\n--- Page {i+1} ---\n{pt}"
            return text if text.strip() else "[PDF appears to be empty or contains only images]"
        except Exception:
            from PyPDF2 import PdfReader as PyPDF2Reader
            pdf = PyPDF2Reader(io.BytesIO(content))
            text = ""
            for i, page in enumerate(pdf.pages):
                pt = page.extract_text()
                if pt:
                    text += f"\n--- Page {i+1} ---\n{pt}"
            return text if text.strip() else "[PDF appears to be empty or contains only images]"
    except Exception as e:
        logger.exception("PDF extraction error")
        return f"[Error extracting PDF text: {str(e)}]"

# --- Prompt building (unchanged) ---
def get_role_info(role_key: str):
    role = JOB_ROLES.get(role_key)
    if role:
        return {"title": role["name"], "focus_areas": ["General DPW operations"]}
    return None

def build_system_prompt(dept_key: str, role_key: Optional[str], whitelist_urls: List[str]) -> str:
    base = DEPARTMENT_PROMPTS.get(dept_key, DEPARTMENT_PROMPTS["general_public_works"]).get("prompt", "")
    role_txt = ""
    if role_key:
        role = get_role_info(role_key)
        if role:
            areas = role.get("focus_areas", [])
            role_txt = f"\n\nROLE CONTEXT:\n- Title: {role.get('title', role_key)}\n- Focus Areas:\n" + "\n".join(f"  - {a}" for a in areas)
    domains = get_whitelisted_domains(whitelist_urls)
    whitelist_notice = (
        f"\n\nURL RESTRICTIONS:\n"
        f"- Only cite and reference sources from approved whitelist\n"
        f"- Include the specific URL for each citation\n"
        f"- If info is not in whitelist, clearly state that it cannot be verified from approved sources\n"
        f"- All child pages of whitelisted URLs are permitted\n"
        f"- Total Whitelisted URLs: {len(whitelist_urls)}\n"
        f"- Approved Domains: {', '.join(sorted(list(domains))[:25])}"
        + ("..." if len(domains) > 25 else "")
    )
    return base + role_txt + whitelist_notice

# --- Gemini client and response generation (largely unchanged) ---
def init_clients():
    timeout_config = httpx.Timeout(connect=90.0, read=240.0, write=90.0, pool=60.0)
    limits_config = httpx.Limits(max_connections=50, max_keepalive_connections=10, keepalive_expiry=30.0)
    try:
        http_client = httpx.Client(timeout=timeout_config, limits=limits_config, verify=certifi.where(),
                                   http2=False, follow_redirects=True,
                                   transport=httpx.HTTPTransport(retries=5, verify=certifi.where()))
    except Exception as e:
        logger.warning(f"SSL verify failed for httpx: {e}. Falling back to verify=False")
        http_client = httpx.Client(timeout=timeout_config, limits=limits_config, verify=False,
                                   http2=False, follow_redirects=True)
    gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
    return http_client, gemini_client

def generate_mock_response(query: str, context: str, system_prompt: str, has_document: bool) -> str:
    department_match = re.search(r'You are a specialized AI assistant for ([\w\s&]+)', system_prompt)
    department_name = department_match.group(1).strip() if department_match else 'N/A'
    mock = """[DEMO MODE - Gemini API key not configured]

Your question: {query}

This is a demonstration response. To get real AI-powered answers:
1. Set the GEMINI_API_KEY environment variable.
2. Rerun the app.

Configuration:
- Department: {department}
- Document uploaded: {doc} (Preview: {preview}...)

*All functionality is ready; needs API key.*"""
    return mock.format(query=query, department=department_name, doc='Yes' if has_document else 'No', preview=context[:50])

def generate_llm_response(query: str, context: str, system_prompt: str, has_document: bool, gemini_client: Optional[genai.Client]) -> str:
    if not gemini_client:
        return generate_mock_response(query, context, system_prompt, has_document)

    contents = [
        {"role": "user", "parts": [{"text": system_prompt}]},
        {"role": "model", "parts": [{"text": "Understood. I will follow the instructions and use the whitelisted sources for citation. I am ready for your query."}]},
        {"role": "user", "parts": [{"text": f"User query: {query}"}, {"text": f"\n\nDOCUMENT CONTEXT (for RAG/citation only): {context[:8000]}" if context else "\n\nNo document uploaded"}]},
    ]

    max_retries = 5
    base_delay = 5.0
    for attempt in range(max_retries):
        try:
            resp = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config={"max_output_tokens": 8192, "temperature": 0.7, "timeout": 240.0}
            )
            if getattr(resp, "candidates", None) and resp.candidates[0].content.parts:
                return resp.text
            fr = getattr(resp.candidates[0], "finish_reason", None)
            if fr and getattr(fr, "name", "") != "STOP":
                return f"[LLM Response Blocked] The model finished generation with reason: {getattr(fr, 'name', 'UNKNOWN')}."
            raise RuntimeError("Empty response from LLM.")
        except APIError as e:
            s = str(e).lower()
            is_timeout = "timeout" in s
            is_connection = is_timeout or "connection" in s or "network" in s
            is_rate = "rate" in s or "429" in s
            is_server = any(code in s for code in ["500", "502", "503", "504"])
            should_retry = is_timeout or is_connection or is_rate or is_server
            if not should_retry or attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            if is_rate:
                delay *= 3
            elif is_timeout or is_connection:
                delay *= 2
            time.sleep(delay + random.uniform(0.1 * delay, 0.3 * delay))
        except Exception:
            if attempt == 0:
                time.sleep(5)
                continue
            raise

# --- Streamlit UI & State ---
if "whitelist" not in st.session_state:
    st.session_state.whitelist = fetch_whitelist()
if "document_context" not in st.session_state:
    st.session_state.document_context = ""
if "questions" not in st.session_state:
    st.session_state.questions = []
if "http_client" not in st.session_state or "gemini_client" not in st.session_state:
    st.session_state.http_client, st.session_state.gemini_client = init_clients()

st.set_page_config(page_title="PipeWrench AI", layout="wide")
st.title("PipeWrench AI - Municipal DPW Knowledge Capture (Streamlit)")

with st.sidebar:
    st.header("Configuration")
    dept = st.selectbox("Department", list(DEPARTMENT_PROMPTS.keys()), index=0, format_func=lambda k: DEPARTMENT_PROMPTS[k]["name"])
    role = st.selectbox("Role", list(JOB_ROLES.keys()), index=0, format_func=lambda k: JOB_ROLES[k]["name"])
    st.text(f"Model: {GEMINI_MODEL}")
    api_key_input = st.text_input("GEMINI_API_KEY (optional if set in env)", type="password")
    if api_key_input and api_key_input != GEMINI_API_KEY:
        GEMINI_API_KEY = api_key_input
        st.session_state.http_client, st.session_state.gemini_client = init_clients()
        st.success("Gemini client reinitialized with provided key.")

    st.divider()
    st.write(f"Whitelisted URLs: {len(st.session_state.whitelist)}")
    st.write("Approved Domains:")
    st.caption(", ".join(sorted(list(get_whitelisted_domains(st.session_state.whitelist)))[:10]) + ("..." if len(get_whitelisted_domains(st.session_state.whitelist)) > 10 else ""))

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ask a question")
    q = st.text_area("Your question", height=120, placeholder="e.g., What are EPA requirements for stormwater BMP inspections?")
    uploaded = st.file_uploader("Upload a PDF for context (optional)", type=["pdf"])
    if uploaded is not None:
        contents = uploaded.read()
        text = extract_text_from_pdf(contents)
        if text.startswith("[Error") or text.startswith("[ERROR"):
            st.error(f"PDF processing failed: {text}")
        else:
            st.session_state.document_context = text
            pages = text.count("--- Page")
            st.success(f"Extracted context from {uploaded.name} ({pages} pages).")

    if st.button("Generate Answer", type="primary", disabled=not q.strip()):
        system_prompt = build_system_prompt(dept, role, st.session_state.whitelist)
        try:
            with st.spinner("Contacting model..."):
                resp = generate_llm_response(
                    query=q.strip(),
                    context=st.session_state.document_context,
                    system_prompt=system_prompt,
                    has_document=bool(st.session_state.document_context),
                    gemini_client=st.session_state.gemini_client
                )
            final = enforce_whitelist_on_text(resp, st.session_state.whitelist)
            st.session_state.questions.append({"q": q.strip(), "a": final, "ts": datetime.now().isoformat()})
            st.success("Answer ready.")
        except Exception as e:
            st.error(f"LLM error: {e}")

    # Display conversation
    for item in reversed(st.session_state.questions[-10:]):
        st.markdown(f"**You:** {item['q']}")
        st.markdown(f"**PipeWrench:**\n\n{item['a']}")
        st.caption(item["ts"])
        st.divider()

with col2:
    st.subheader("Diagnostics")
    if st.button("Test DNS (Gemini)"):
        try:
            ip = socket.gethostbyname("api.gemini.google.com")
            st.info(f"DNS OK: api.gemini.google.com -> {ip}")
        except Exception as e:
            st.error(f"DNS failed: {e}")

    if st.button("HTTP GET https://api.gemini.google.com"):
        try:
            r = requests.get("https://api.gemini.google.com", timeout=10)
            st.info(f"requests status: {r.status_code}")
        except Exception as e:
            st.error(f"requests failed: {e}")

    if st.button("HTTPX verify=True"):
        try:
            with httpx.Client(timeout=10.0, verify=True) as client:
                r = client.get("https://api.gemini.google.com")
                st.info(f"httpx status: {r.status_code}")
        except Exception as e:
            st.error(f"httpx failed: {str(e)[:200]}")

    st.subheader("Session")
    st.text(f"Context length: {len(st.session_state.document_context)} chars")
    st.text(f"Questions: {len(st.session_state.questions)}")
