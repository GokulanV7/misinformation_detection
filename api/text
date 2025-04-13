# api/main.py

import logging
import time
import asyncio
import uuid
import os
import re
import json # Required for synthesis parsing
from contextlib import asynccontextmanager
from typing import Dict, Optional, List, Literal, Any, Union, Tuple

import httpx # Keep httpx for potential use
# Need NetworkX for the graph check in status endpoint
import networkx as nx
from fastapi import FastAPI, HTTPException, Depends, Request, Header, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from redis import asyncio as aioredis
from dotenv import load_dotenv

# --- Import local modules ---
try:
    from .models import (
        AnalyzeRequest, BaseAnalysisResponse, FactualAnalysisResponse, MisinformationAnalysisResponse,
        UrlAnalysisResponse, StatusResponse, ErrorResponse, TextContextAssessment, ScanResultDetail,
        UrlScanResults, EvidenceItem # Ensure EvidenceItem is imported
    )
    from .classifier import classify_intent, load_classifier
    from .groq_utils import (
        query_groq, setup_groq_client, close_groq_client, GroqApiException,
        ask_groq_factual, analyze_misinformation_groq # Ensure specific utils are imported
    )
    from .langchain_utils import RealTimeDataProcessor # Handles RAG + Cohere
    from .utils import get_config, setup_logging, is_valid_url, RateLimitException, ApiException, sanitize_url_for_scan
    from .vt_utils import check_virustotal, parse_vt_result
    from .ipqs_utils import check_ipqs, parse_ipqs_result
    from .urlscan_utils import check_urlscan_existing_results, parse_urlscan_result
    from .search_api_utils import perform_search # Required for web search fallback
    from .kg_utils import load_graph, save_graph, load_spacy_model, extract_entities, add_claim_to_graph, query_kg_for_entities, graph as kg_global_graph # Import global graph for status check
except ImportError as e:
     print(f"ERROR: Failed to import necessary modules: {e}")
     print("Ensure all util files exist and Python can find the 'api' package.")
     exit(1)


# --- Basic Setup ---
setup_logging()
logger = logging.getLogger(__name__)
load_dotenv()
CONFIG = get_config()

# --- Configuration Constants ---
ANALYSIS_CONFIG = CONFIG.get('analysis', {})
WEB_FALLBACK_THRESHOLD = ANALYSIS_CONFIG.get('web_fallback_threshold', 0.70)
CACHE_TIMEOUT = CONFIG.get('cache', {}).get('default_ttl_seconds', 300)
API_KEY_ENABLED = CONFIG.get("security", {}).get("enable_api_key_auth", False)
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")
API_KEY_NAME = "X-API-Key"

# --- Globals (initialized in lifespan) ---
rag_processor: Optional[RealTimeDataProcessor] = None
redis_client: Optional[aioredis.Redis] = None # Hold redis client ref for shutdown
shutdown_event = asyncio.Event()

# --- API Key Security ---
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(key: str = Security(api_key_header)):
    """Dependency to validate the API Key if security is enabled."""
    if not API_KEY_ENABLED:
        logger.debug("API Key auth disabled.")
        return None
    if not key:
         logger.info("Request rejected: API Key required via X-API-Key header.")
         raise HTTPException(status_code=401, detail={"error": "Unauthorized", "message": "API Key required via X-API-Key header."})
    if not INTERNAL_API_KEY:
         logger.error("API Key security enabled, but INTERNAL_API_KEY not set. Denying request.")
         raise HTTPException(status_code=500, detail={"error": "Configuration Error", "message": "API Key authentication is misconfigured on the server."})
    if key == INTERNAL_API_KEY:
        logger.debug("API Key validated successfully.")
        return key
    else:
        logger.warning("Request rejected: Invalid API Key provided.")
        raise HTTPException(status_code=401, detail={"error": "Unauthorized", "message": "Invalid API Key."})

# --- App Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_processor, redis_client
    logger.info("Application startup initiated...")

    # 1. Setup HTTP client for Groq utils
    setup_groq_client()

    # 2. Initialize Cache
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    logger.info(f"Connecting to Redis at {redis_url} for caching...")
    try:
        redis_client = aioredis.from_url(redis_url, encoding="utf8", decode_responses=True)
        await redis_client.ping()
        FastAPICache.init(RedisBackend(redis_client), prefix="fastapi-cache")
        logger.info("Redis connection successful. FastAPI Cache initialized.")
    except Exception as e:
        logger.error(f"Failed to connect to Redis or initialize cache: {e}. Cache will be disabled.", exc_info=True)
        redis_client = None # Ensure client is None if failed

    # 3. Load Classifier
    logger.info("Loading intent classifier model...")
    if not load_classifier():
        logger.error("Failed to load intent classifier model. Startup aborted.")
        raise RuntimeError("Failed to load essential intent classifier model.")
    logger.info("Intent classifier loaded.")

    # 4. Initialize RAG Processor
    logger.info("Initializing RAG processor...")
    try:
        rag_processor = RealTimeDataProcessor()
        if not rag_processor or not rag_processor.embeddings or not rag_processor.vector_store:
             logger.warning("RAG processor failed initial setup (embeddings or vector store missing/failed). RAG functionality will be limited.")
             # If RAG is essential, raise error here
             # raise RuntimeError("Failed to initialize essential RAG processor.")
        else:
             logger.info("RAG processor initialized successfully.")
    except Exception as e:
         logger.error(f"Critical error initializing RAG processor: {e}", exc_info=True)
         rag_processor = None # Mark as unavailable
         # raise RuntimeError(f"Critical error initializing RAG processor: {e}")

    # 5. Load Knowledge Graph & SpaCy
    logger.info("Loading Knowledge Graph...")
    try: load_graph()
    except Exception as e: logger.error(f"Failed to load knowledge graph: {e}", exc_info=True) # Non-fatal

    logger.info("Loading SpaCy model for KG NER...")
    try:
        if not load_spacy_model(): logger.warning("Failed to load SpaCy model. KG entity extraction disabled.") # Non-fatal
    except Exception as e: logger.error(f"Error loading SpaCy model: {e}", exc_info=True)


    logger.info("Application startup complete.")
    yield  # API is now running

    # --- Shutdown Sequence ---
    logger.info("Application shutdown initiated...")
    shutdown_event.set()

    # Graceful shutdown tasks
    try: save_graph()
    except Exception as e: logger.error(f"Error saving graph on shutdown: {e}")

    await close_groq_client() # Close shared client

    if redis_client:
        try:
            await redis_client.close()
            await redis_client.connection_pool.disconnect() # Ensure pool is disconnected too
            logger.info("Redis connection closed.")
        except Exception as e: logger.error(f"Error closing Redis connection: {e}")

    logger.info("Application shutdown complete.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Hack the Hoax - Misinformation Detector API",
    description="Advanced API using RAG, LLMs, multiple URL scanners, KG, and web search synthesis for real-time analysis.",
    version="1.2.0", # Incremented version for synthesis update
    lifespan=lifespan,
    # Add custom error response schema for validation errors if desired
    # responses={422: {"model": ErrorResponse, "description": "Validation Error"}}
)

# --- CORS Configuration ---
allowed_origins = CONFIG.get("api", {}).get("cors_allowed_origins", ["*"])
logger.info(f"Configuring CORS for origins: {allowed_origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Helper Functions (URL Analysis, Context, Scans, Synthesis, Assessment) ---

def _analyze_text_context(request_id: str, context_text: str) -> TextContextAssessment:
    """Analyzes surrounding text for scam/phishing keywords."""
    logger.debug(f"[ReqID: {request_id}] Analyzing text context: '{context_text[:100]}...'")
    suspicion: Literal["High", "Medium", "Low"] = "Low"; indicators = []
    lower_context = context_text.lower()
    # Keywords list - could be moved to config if large/dynamic
    high_kws = ["free", "won", "prize", "lottery", "claim", "urgent", "verify", "login", "update", "suspended", "confirm identity", "password", "ssn", "bank", "payment", "tax refund", "irs", "action required"]
    med_kws = ["delivery", "package", "confirm", "notification", "invoice", "shipment", "alert", "security", "gift card", "refund", "support", "account issue", "survey", "exclusive offer"]

    matched_h = [k for k in high_kws if k in lower_context]
    matched_m = [k for k in med_kws if k in lower_context]

    # Prioritize high suspicion keywords
    if matched_h:
        suspicion = "High"
        indicators = sorted(list(set(matched_h))) # Use unique sorted list
        logger.debug(f"Text context assessment: High (Keywords: {indicators})")
    elif matched_m:
        suspicion = "Medium"
        indicators = sorted(list(set(matched_m))) # Use unique sorted list
        logger.debug(f"Text context assessment: Medium (Keywords: {indicators})")
    else:
         logger.debug("Text context assessment: Low (No significant keywords)")

    return TextContextAssessment(suspicion_level=suspicion, key_indicators=indicators)

async def _perform_url_scans(request_id: str, url: str) -> Dict[str, Optional[ScanResultDetail]]:
    """Runs VT, IPQS, URLScan concurrently and processes results."""
    logger.info(f"[ReqID: {request_id}] Performing concurrent scans for URL: {url}")
    # Ensure URL is sanitized before sending to scanners
    sanitized_url = sanitize_url_for_scan(url)
    if not sanitized_url or not is_valid_url(sanitized_url):
        logger.error(f"[ReqID: {request_id}] Invalid or unsanitizable URL for scanning: {url}")
        raise ValueError("Invalid URL provided for scanning.")

    # Define tasks for each scanner
    scan_tasks = {
        "virustotal": check_virustotal(sanitized_url),
        "ipqualityscore": check_ipqs(sanitized_url),
        "urlscanio": check_urlscan_existing_results(sanitized_url) # Check existing scans
    }
    parser_map = {
        "virustotal": parse_vt_result,
        "ipqualityscore": parse_ipqs_result,
        "urlscanio": parse_urlscan_result
    }

    scan_start = time.perf_counter()
    # Execute tasks concurrently, capturing results or exceptions
    results = await asyncio.gather(*scan_tasks.values(), return_exceptions=True)
    scan_duration = time.perf_counter() - scan_start
    logger.debug(f"[ReqID: {request_id}] URL scans completed in {scan_duration:.3f}s")

    scan_outputs: Dict[str, Optional[ScanResultDetail]] = {}
    any_critical_failure = False # Flag if a core scanner fails unexpectedly

    # Process results for each scanner
    for i, (scanner_name, result_or_exc) in enumerate(zip(scan_tasks.keys(), results)):
        status = "error"; details = {"message": "Unknown processing error"}

        if isinstance(result_or_exc, RateLimitException):
            status = "rate_limited"; details = {"message": str(result_or_exc)}
            logger.warning(f"[ReqID: {request_id}] Scanner '{scanner_name}' hit rate limit for URL: {sanitized_url}")
        elif isinstance(result_or_exc, ApiException):
            status = "error"; details = {"message": str(result_or_exc)}
            logger.error(f"[ReqID: {request_id}] Scanner '{scanner_name}' API error for URL {sanitized_url}: {result_or_exc}")
            any_critical_failure = True
        elif isinstance(result_or_exc, ValueError): # e.g., invalid URL for a specific scanner
             status = "error"; details = {"message": f"Input error: {result_or_exc}"}
             logger.warning(f"[ReqID: {request_id}] Scanner '{scanner_name}' input error for URL {sanitized_url}: {result_or_exc}")
        elif isinstance(result_or_exc, Exception): # Catch other unexpected exceptions
            status = "error"; details = {"message": f"Unexpected internal error: {type(result_or_exc).__name__}"}
            logger.error(f"[ReqID: {request_id}] Unexpected error in scanner '{scanner_name}' for URL {sanitized_url}: {result_or_exc}", exc_info=True)
            any_critical_failure = True
        elif result_or_exc is None: # Scanner explicitly returned None (e.g., VT 404, IPQS timeout/no_data, urlscan no result)
            status = "no_data"; details = {"message": f"{scanner_name} reported no specific data found."}
            logger.info(f"[ReqID: {request_id}] Scanner '{scanner_name}' found no existing data for URL: {sanitized_url}")
        else: # Success case - raw data received, attempt parsing
            parser = parser_map.get(scanner_name)
            if parser:
                try:
                    parsed = parser(result_or_exc) # result_or_exc should be the data dict here
                    status = parsed.get("status", "error") # Parser should return status ('success', 'no_scan_found', 'pending', etc.)
                    details = parsed.get("details", {})
                    if status == "error": # If parser itself encountered an error
                        logger.warning(f"[ReqID: {request_id}] Parser for '{scanner_name}' failed or returned error status for URL {sanitized_url}. Details: {details.get('message', 'N/A')}")
                    elif status not in ["success", "scan_found", "no_scan_found", "pending", "likely_safe"]: # Allow likely_safe from parser
                         logger.warning(f"[ReqID: {request_id}] Parser for '{scanner_name}' returned unexpected status '{status}' for URL {sanitized_url}")

                except Exception as e:
                    status = "error"; details = {"message": f"Failed to parse {scanner_name} response: {e}"}
                    logger.error(f"[ReqID: {request_id}] Exception parsing '{scanner_name}' result for URL {sanitized_url}: {e}", exc_info=True)
                    any_critical_failure = True
            else:
                status = "error"; details = {"message": f"Internal error: No parser defined for {scanner_name}"}
                logger.error(f"[ReqID: {request_id}] Missing parser definition for scanner: {scanner_name}")
                any_critical_failure = True

        # Store the parsed result detail
        scan_outputs[scanner_name] = ScanResultDetail(status=status, details=details)

    # Check if ALL scans failed critically (not just 'no_data' or 'rate_limited')
    critical_failures = [s for s in scan_outputs.values() if s and s.status == 'error']
    if len(critical_failures) == len(scan_tasks):
         # If all scanners resulted in an error state, raise a general API exception
         error_messages = "; ".join([f"{name}: {res.details.get('message', 'Unknown error')}" for name, res in scan_outputs.items() if res])
         logger.error(f"[ReqID: {request_id}] All URL scanners failed with errors for URL {sanitized_url}: {error_messages}")
         raise ApiException(f"Critical failure: All URL scanning services returned errors. Details: {error_messages}")

    return scan_outputs

def consolidate_url_assessment(
    url: str, # The original *unsanitized* URL for display, or "N/A"
    sanitized_url: Optional[str], # The URL actually scanned
    text_assessment: Optional[TextContextAssessment],
    scan_outputs: Dict[str, Optional[ScanResultDetail]]
) -> Dict[str, Any]:
    """Consolidates text and scanner results into a final URL assessment dict."""
    # Use provided request_id if possible, else None
    request_id = getattr(asyncio.current_task(), 'request_id', None) # Hacky way to maybe get ID if set on task local
    log_prefix = f"[ReqID: {request_id}] " if request_id else ""
    logger.debug(f"{log_prefix}Consolidating URL assessment for: {url}")

    evidence_notes = []
    confidence = 0.5 # Base confidence, adjusted by findings
    AssessmentType = Literal["Malicious", "Phishing", "Spam", "Suspicious", "Likely Safe", "Uncertain", "Analysis Failed"]
    assessment: AssessmentType = "Uncertain" # Default
    analysis_summary = "Analysis initiated."

    malicious_score = 0.0; phishing_score = 0.0; spam_score = 0.0; suspicion_score = 0.0; safety_score = 0.0
    scanner_success_count = 0

    # --- Text Context Processing ---
    text_suspicion = "N/A"
    if text_assessment:
        text_suspicion = text_assessment.suspicion_level
        if text_suspicion == "High":
            suspicion_score += 3.0; spam_score += 1.0; evidence_notes.append("Text context flagged as highly suspicious.")
        elif text_suspicion == "Medium":
            suspicion_score += 1.5; evidence_notes.append("Text context flagged as medium suspicion.")
        if text_assessment.key_indicators:
            evidence_notes.extend([f"Text Indicator: '{ind}'" for ind in text_assessment.key_indicators])

    # --- Scanner Results Processing ---
    vt_res = scan_outputs.get('virustotal')
    ipqs_res = scan_outputs.get('ipqualityscore')
    urlscan_res = scan_outputs.get('urlscanio')

    # Process VirusTotal
    if vt_res and vt_res.status == 'success' and vt_res.details:
        scanner_success_count += 1
        d = vt_res.details
        pos = d.get('positives', 0); total = d.get('total_engines', 0); assess = d.get('assessment', 'N/A')
        evidence_notes.append(f"VirusTotal: {pos}/{total} engines flagged ({assess}). Rep: {d.get('reputation', 0)}.")
        if assess == 'malicious': malicious_score += 3.0 + pos * 0.1; suspicion_score += 2.0
        elif assess == 'suspicious': suspicion_score += 1.5 + pos * 0.15
        elif assess == 'likely_safe': safety_score += 2.0
    elif vt_res and vt_res.status not in ['no_data', 'skipped']: # Log errors/rate limits etc.
        evidence_notes.append(f"VirusTotal Status: {vt_res.status} ({vt_res.details.get('message', 'No details') if vt_res.details else 'N/A'}).")

    # Process IPQualityScore
    if ipqs_res and ipqs_res.status == 'success' and ipqs_res.details:
        scanner_success_count += 1
        d = ipqs_res.details
        risk = d.get('risk_score', -1); threat = d.get('threat_category', 'N/A'); is_p = d.get('is_phishing'); is_m = d.get('is_malware'); is_s = d.get('is_spam'); age_desc = d.get('domain_age_description', 'N/A')
        evidence_notes.append(f"IPQS: Risk {risk}, Cat '{threat}', Phish:{is_p}, Mal:{is_m}, Spam:{is_s}. Age: {age_desc}.")
        if is_p: phishing_score += 6.0; suspicion_score += 2.0
        if is_m: malicious_score += 5.0; suspicion_score += 1.0
        if is_s: spam_score += 3.0; suspicion_score += 0.5 # Spam adds to suspicion too
        if d.get('assessment_category') == 'high_risk': suspicion_score += 4.0
        elif d.get('assessment_category') == 'medium_risk': suspicion_score += 2.0
        elif d.get('assessment_category') == 'low_risk': safety_score += 1.0
        # Check if age indicates recent (less than ~90 days)
        is_young = False
        if isinstance(age_desc, str):
             age_num_match = re.search(r'\d+', age_desc)
             if age_num_match:
                  try:
                      age_num = int(age_num_match.group())
                      if 'day' in age_desc.lower() and age_num < 90: is_young = True
                      if 'month' in age_desc.lower() and age_num < 3: is_young = True
                  except ValueError: pass # Ignore if number extraction fails
        if is_young: suspicion_score += 1.0; evidence_notes.append("Note: Domain age appears recent (<3 months).")
    elif ipqs_res and ipqs_res.status not in ['no_data', 'skipped']:
        evidence_notes.append(f"IPQualityScore Status: {ipqs_res.status} ({ipqs_res.details.get('message', 'No details') if ipqs_res.details else 'N/A'}).")

    # Process Urlscan.io
    if urlscan_res and urlscan_res.status == 'scan_found' and urlscan_res.details:
        scanner_success_count += 1
        d = urlscan_res.details
        tags = d.get('tags', []); score = d.get('score',0); is_malicious_verdict = d.get('verdict_malicious')
        tags_str = ', '.join(tags) if tags else 'None'; evidence_notes.append(f"URLScan.io: Found scan ({d.get('scan_date', 'N/A')[:10]}), Score:{score}, Malicious:{is_malicious_verdict}. Tags:[{tags_str[:50]}...]")
        if is_malicious_verdict: malicious_score += 5.0; suspicion_score += 2.0
        elif d.get('assessment_category') == 'suspicious': suspicion_score += 3.0
        suspicious_tags = ['phishing', 'malware', 'suspicious', 'crypto-scam', 'hacked', 'dynamic dns', 'spam']
        if any(t in tags for t in suspicious_tags): suspicion_score += 2.5
        if any(t == 'phishing' for t in tags): phishing_score += 3.0

        if score > 75: suspicion_score += 1.5
        elif score < 10: safety_score += 0.5
    elif urlscan_res and urlscan_res.status == 'no_scan_found':
        evidence_notes.append("URLScan.io: No existing scan found for this domain.")
    elif urlscan_res and urlscan_res.status not in ['skipped']: # Log errors, rate limits etc.
        evidence_notes.append(f"URLScan.io Status: {urlscan_res.status} ({urlscan_res.details.get('message', 'No details') if urlscan_res.details else 'N/A'}).")

    # --- Final Assessment Logic ---
    PHISHING_CONFIRMED_THR = 5.5; MALICIOUS_CONFIRMED_THR = 5.0; SUSPICIOUS_THR = 3.5; SPAM_THR = 2.5; SAFE_THR = 1.5
    net_safety_score = (safety_score * (1 + scanner_success_count * 0.1)) - suspicion_score

    if phishing_score >= PHISHING_CONFIRMED_THR:
        assessment = "Phishing"; confidence = min(0.9 + phishing_score * 0.01, 0.99); analysis_summary = f"High risk of phishing detected based on scanner results."
    elif malicious_score >= MALICIOUS_CONFIRMED_THR:
        assessment = "Malicious"; confidence = min(0.9 + malicious_score * 0.01, 0.99); analysis_summary = f"High risk of malicious activity detected."
    elif spam_score >= SPAM_THR and text_suspicion in ["High", "Medium"]:
        assessment = "Spam"; confidence = min(0.7 + spam_score * 0.05 + (0.1 if text_suspicion == "High" else 0), 0.90); analysis_summary = "URL potentially related to spam, supported by text context."
    elif suspicion_score >= SUSPICIOUS_THR or (suspicion_score > 1.5 and text_suspicion == "High"):
        assessment = "Suspicious"; confidence = min(0.55 + suspicion_score * 0.06 + (0.1 if text_suspicion == "High" else 0), 0.88); analysis_summary = "URL flagged as potentially suspicious based on scan results and/or text context."
    elif net_safety_score >= SAFE_THR and scanner_success_count > 0 :
        assessment = "Likely Safe"; confidence = min(0.75 + safety_score * 0.05, 0.95); analysis_summary = "URL appears likely safe based on available scan results."
    else: # Default to Uncertain
        assessment = "Uncertain"; confidence = max(0.4 - suspicion_score * 0.1 + safety_score * 0.05, 0.2); analysis_summary = "Analysis inconclusive or requires further verification. Limited or conflicting signals."

    # Handle cases where no URL was provided or scanning failed critically
    if url == "N/A" or not sanitized_url:
        assessment = "Analysis Failed"; confidence = 0.1; analysis_summary = "No processable URL found in the input text."; evidence_notes = ["No processable URL identified."]
        scan_outputs = {s: ScanResultDetail(status="skipped", details={"message":"No URL found"}) for s in ["virustotal","ipqualityscore","urlscanio"]}
    elif scanner_success_count == 0 and url != "N/A": # URL existed but NO scans succeeded
        assessment = "Analysis Failed"; confidence = 0.1; analysis_summary = "Could not retrieve results from external URL scanners due to errors or timeouts."
        evidence_notes.append("Failed to get conclusive results from scanning services.")

    # Ensure scan_details structure is correct for the response model
    final_scan_results = UrlScanResults(
        virustotal=scan_outputs.get('virustotal'), ipqualityscore=scan_outputs.get('ipqualityscore'), urlscanio=scan_outputs.get('urlscanio')
    )

    return {
        "assessment": assessment, "scanned_url": sanitized_url or url, "confidence_score": round(confidence, 3),
        "analysis_summary": analysis_summary, "text_context_assessment": text_assessment, "scan_results": final_scan_results,
        "evidence_notes": evidence_notes
    }


async def _synthesize_from_web_results(
    request_id: str,
    original_query: str,
    search_results: List[Dict[str, str]],
    analysis_type: Literal["misinfo", "factual"]
) -> Dict[str, Any]:
    """
    Uses Groq to synthesize a consolidated answer/assessment based on web search results.

    Returns a dictionary containing updated fields:
    { "assessment": ..., "confidence_score": ..., "answer": ..., "explanation": ...,
      "evidence": ..., "data_source": ..., "raw_llm_output": ... }
    The 'answer' field (for factual) or 'explanation' field (for misinfo) will contain the synthesized narrative.
    """
    logger.info(f"[ReqID: {request_id}] Synthesizing response from {len(search_results)} web results for type: {analysis_type}")
    if not search_results:
        logger.warning(f"[ReqID: {request_id}] No search results provided for synthesis.")
        return {"explanation": "Web search returned no results for synthesis.", "answer": None, "assessment": "Needs Verification / Uncertain", "confidence_score": 0.1, "data_source": "Web Search", "evidence": []}

    snippets_text = "\n\n---\n".join(
        [f"Source URL: {res.get('link', 'N/A')}\nTitle: {res.get('title', 'N/A')}\nSnippet: {res.get('snippet', 'N/A')}" for res in search_results]
    )
    max_snippet_length = 3500
    if len(snippets_text) > max_snippet_length:
        logger.warning(f"[ReqID: {request_id}] Truncating web snippets from {len(snippets_text)} to {max_snippet_length} chars for LLM prompt.")
        snippets_text = snippets_text[:max_snippet_length] + "... (truncated)"


    synthesis_model = 'llama3-70b-8192' # Powerful model for quality synthesis
    prompt = ""
    temp = 0.15 # Slightly increased temp for better narrative flow, adjust if needed

    # --- PROMPT REFINEMENT ---
    if analysis_type == "factual":
        prompt = f"""You are an expert factual answer synthesizer. Your goal is to provide a single, comprehensive, and neutral answer to the user's question, based *only* on the information contained in the provided web search snippets.

Instructions:
1. Carefully read the user's question.
2. Analyze each web search snippet provided below for relevance to the question.
3. Synthesize the information from the *relevant* snippets into a single, coherent paragraph or short text.
4. Focus on presenting the facts found in the snippets. Do *not* include opinions or information not present in the snippets.
5. If the snippets provide conflicting information, briefly note the contradiction.
6. If the snippets do not contain enough information to fully answer the question, state what *can* be concluded from the snippets and explicitly mention what information is missing or couldn't be verified from the provided text.
7. Structure your response as a direct answer to the question. Start the answer directly, without introductory phrases like "Based on the snippets...".

User Question: "{original_query}"

Web Search Snippets:
{snippets_text}

---
Synthesized Answer (Based ONLY on the snippets):"""

    elif analysis_type == "misinfo":
        temp = 0.25 # Slightly higher temp for nuanced explanation
        prompt = f"""You are an expert misinformation analyst. Your task is to analyze the user's statement based *only* on the provided web search snippets and provide a detailed explanation consolidating the findings.

Instructions:
1. Read the user's statement carefully.
2. Analyze each web snippet to see how it relates to the statement (supports, contradicts, provides context, irrelevant).
3. Synthesize these findings into a detailed explanation. Explicitly mention which snippets support which points, if possible. Highlight any contradictions or nuances found in the snippets.
4. Conclude your analysis by assessing whether the snippets overall lean towards the statement being factual, misleading, opinion-based, contradictory, or if the snippets are insufficient.
5. Return your response *only* as a single JSON object enclosed in ```json ... ``` with the following keys:
    "assessment": (Choose ONE based *only* on the snippets: "Likely Factual based on snippets", "Likely Misleading based on snippets", "Opinion based on snippets", "Insufficient information in snippets", "Contradictory information in snippets")
    "confidence": (float, 0.0-1.0, your confidence *in the assessment derived solely from the provided snippets*)
    "synthesized_explanation": (string, **detailed analysis and consolidation** of how the snippets relate to the original statement. Explain the reasoning for the assessment, citing snippet information.)

User Statement: "{original_query}"

Web Search Snippets:
{snippets_text}

---
```json
{{ ... }}
```"""

    else: # Should not happen
        logger.error(f"[ReqID: {request_id}] Invalid analysis_type '{analysis_type}' for synthesis.")
        return {"explanation": "Internal error: Invalid synthesis type.", "answer":None, "assessment": "Needs Verification / Uncertain", "confidence_score": 0.0, "data_source": "N/A", "evidence": []}

    # --- Call Groq for Synthesis ---
    try:
        logger.info(f"[ReqID: {request_id}] Sending synthesis request to Groq model {synthesis_model} (Temp: {temp}).")
        synthesis_llm_output = await query_groq(prompt, temperature=temp, model=synthesis_model, max_tokens=1536) # Increased max tokens for narrative

        if not synthesis_llm_output:
            logger.warning(f"[ReqID: {request_id}] Web synthesis LLM call returned empty.")
            return {"explanation": "Web search synthesis failed (LLM returned empty).", "answer": None, "assessment": "Needs Verification / Uncertain", "confidence_score": 0.15, "data_source": "Web Search", "evidence": []}

        # --- Parse the Synthesis Result ---
        synthesized_data = {"raw_llm_output": synthesis_llm_output, "answer": None, "explanation": None}
        # Prepare evidence list from the search results *used* for synthesis
        web_evidence = [EvidenceItem(source=res.get('link', 'N/A'), snippet=res.get('snippet', 'N/A')[:300]+"...", assessment_note="Used in Web Search Synthesis") for res in search_results]
        synthesized_data["evidence"] = web_evidence # Include evidence used

        if analysis_type == "factual":
            # The entire cleaned output is the synthesized answer
            synthesized_answer = synthesis_llm_output.strip()
            synthesized_data["answer"] = synthesized_answer
            synthesized_data["explanation"] = None # Factual usually doesn't need separate explanation field populated here

            # Re-assess the synthesized answer for uncertainty markers
            lower_answer = synthesized_answer.lower()
            uncertainty_markers = ["cannot answer", "not found", "unable to determine", "snippets do not contain", "don't provide", "insufficient information", "no definitive answer", "missing", "unclear from the snippets"]
            if any(marker in lower_answer for marker in uncertainty_markers):
                synthesized_data["assessment"] = "Needs Verification / Uncertain"
                synthesized_data["confidence_score"] = 0.45 # Slightly higher than pure unknown, but still low
            else:
                synthesized_data["assessment"] = "Likely Factual"
                synthesized_data["confidence_score"] = 0.89 # High confidence if synthesized without explicit uncertainty
            synthesized_data["data_source"] = "Web Search Synthesis"
            logger.info(f"[ReqID: {request_id}] Factual synthesis successful. Assessment: {synthesized_data['assessment']}, Conf: {synthesized_data['confidence_score']:.2f}")

        elif analysis_type == "misinfo":
            json_result = None
            try:
                json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', synthesis_llm_output, re.IGNORECASE | re.DOTALL)
                if json_match: json_str = json_match.group(1); json_result = json.loads(json_str)
                else: # Fallback
                    json_start = synthesis_llm_output.find("{"); json_end = synthesis_llm_output.rfind("}") + 1
                    if json_start != -1 and json_end > json_start: json_result = json.loads(synthesis_llm_output[json_start:json_end])

                # Check for all required keys in the new structure
                if json_result and isinstance(json_result, dict) and all(k in json_result for k in ["assessment", "confidence", "synthesized_explanation"]):
                    assessment_map = { # Map LLM assessment strings to our standard literals
                        "Likely Factual based on snippets": "Likely Factual", "Likely Misleading based on snippets": "Likely Misleading",
                        "Opinion based on snippets": "Opinion", "Insufficient information in snippets": "Needs Verification / Uncertain",
                        "Contradictory information in snippets": "Contradictory Information Found",
                    }
                    llm_assessment_str = json_result.get("assessment")
                    synthesized_data["assessment"] = assessment_map.get(llm_assessment_str, "Needs Verification / Uncertain")
                    synthesized_data["confidence_score"] = min(max(float(json_result.get("confidence", 0.5)) * 0.95, 0.1), 0.95)
                    # Use the detailed synthesized explanation
                    synthesized_data["explanation"] = json_result.get("synthesized_explanation", "Synthesis explanation missing from LLM response.").strip()
                    synthesized_data["answer"] = None # No separate 'answer' for misinfo
                    synthesized_data["data_source"] = "Web Search Synthesis"
                    logger.info(f"[ReqID: {request_id}] Parsed JSON synthesis for misinfo. Assessment: {synthesized_data['assessment']}, Conf: {synthesized_data['confidence_score']:.2f}")

                    # Ensure consistency if LLM flags insufficient info
                    if synthesized_data["assessment"] != "Needs Verification / Uncertain" and llm_assessment_str == "Insufficient information in snippets":
                        logger.warning(f"[ReqID: {request_id}] LLM indicated insufficient info but category mismatch. Overriding assessment.")
                        synthesized_data["assessment"] = "Needs Verification / Uncertain"
                        synthesized_data["confidence_score"] = max(synthesized_data["confidence_score"] * 0.8, 0.3)

                else:
                    raise ValueError(f"Parsed JSON missing required keys (assessment, confidence, synthesized_explanation) or is not a dictionary. Parsed: {json_result}")

            except (json.JSONDecodeError, ValueError) as json_err:
                logger.error(f"[ReqID: {request_id}] Failed to parse valid/complete JSON from web synthesis (misinfo): {json_err}. Raw: '{synthesis_llm_output[:300]}...'")
                synthesized_data["explanation"] = f"Web synthesis ran, but failed to parse structured output from LLM. Raw Output: {synthesis_llm_output}"
                synthesized_data["assessment"] = "Needs Verification / Uncertain"; synthesized_data["confidence_score"] = 0.2
                synthesized_data["data_source"] = "Web Search Synthesis" # Still based on web

        return synthesized_data

    except (RateLimitException, GroqApiException, ApiException) as api_err:
        logger.error(f"[ReqID: {request_id}] API error during web synthesis LLM call: {api_err}")
        return {"explanation": f"Web search synthesis failed due to API error: {api_err}", "answer":None, "assessment": "Needs Verification / Uncertain", "confidence_score": 0.1, "data_source": "Web Search", "evidence": []}
    except Exception as e:
        logger.error(f"[ReqID: {request_id}] Unexpected error during web synthesis: {e}", exc_info=True)
        return {"explanation": f"Unexpected error during web synthesis: {e}", "answer":None, "assessment": "Needs Verification / Uncertain", "confidence_score": 0.1, "data_source": "Web Search", "evidence": []}


def _assess_llm_response_keywords(
    text: str,
    base_confidence: float = 0.6
    ) -> Tuple[Literal["Likely Factual", "Likely Misleading", "Opinion", "Needs Verification / Uncertain", "Contradictory Information Found"], float]:
    """Quick assessment based on keywords in LLM explanation/synthesis. Less critical now synthesis provides its own assessment."""
    lower_text = text.lower() if text else ""
    if not lower_text: return "Needs Verification / Uncertain", 0.1 # Handle empty input

    confidence_bonus = 0.0
    assessment: Literal["Likely Factual", "Likely Misleading", "Opinion", "Needs Verification / Uncertain", "Contradictory Information Found"] = "Needs Verification / Uncertain" # Default

    # Keywords - Less emphasis needed if synthesis step is robust
    misleading_kws = ["contradicts", "inaccurate", "false", "misleading information", "refuted by", "not supported by evidence", "debunked", "untrue"]
    contradictory_kws = ["conflicting information", "contradictory evidence", "disagreement among sources", "mixed results", "both sides", "differing accounts"]
    factual_kws = ["supported by", "accurate statement", "confirmed by", "factual information", "aligns with sources", "evidence suggests", "widely accepted"]
    opinion_kws = ["opinion", "subjective", "viewpoint", "belief", "speculation", "arguable", "interpretation"]
    uncertain_kws = ["insufficient information", "cannot determine", "not addressed", "unclear", "needs more context", "unable to verify", "lacks evidence", "no consensus", "speculative"]
    satire_kws = ["satire", "parody", "humor", "exaggeration", "not serious"]

    if any(w in lower_text for w in misleading_kws): assessment = "Likely Misleading"; confidence_bonus = 0.25
    elif any(w in lower_text for w in contradictory_kws): assessment = "Contradictory Information Found"; confidence_bonus = 0.15
    elif any(w in lower_text for w in factual_kws): assessment = "Likely Factual"; confidence_bonus = 0.25
    elif any(w in lower_text for w in satire_kws): assessment = "Opinion"; confidence_bonus = 0.10
    elif any(w in lower_text for w in opinion_kws): assessment = "Opinion"; confidence_bonus = 0.10
    elif any(w in lower_text for w in uncertain_kws): assessment = "Needs Verification / Uncertain"; confidence_bonus = -0.15
    elif assessment == "Needs Verification / Uncertain": confidence_bonus = 0.0

    final_confidence = min(max(base_confidence + confidence_bonus, 0.05), 0.99)
    return assessment, final_confidence


# --- API Endpoints ---

@app.get("/status", response_model=StatusResponse, tags=["General"])
async def get_status():
    """Provides the operational status of the API and its components."""
    rag_status = "Unavailable"
    if rag_processor and rag_processor.embeddings and rag_processor.vector_store:
         rag_status = "Operational"
    elif rag_processor: rag_status = "Degraded/Partially Initialized"

    kg_status = "Unavailable/Load Failed"
    # Use the globally loaded graph variable reference from kg_utils
    if kg_global_graph is not None and isinstance(kg_global_graph, nx.DiGraph):
         kg_status = "Operational"
    elif kg_global_graph is None: # Explicitly check None if load attempt was made but failed
         kg_status = "Unavailable/Load Failed"
    # If loading was never attempted or module failed import, kg_global_graph might not exist or be None

    cls_status = "Unavailable/Load Failed"
    # Check the global pipeline variable in classifier.py directly
    # Need to import it or the check function `load_classifier` itself needs adjustment
    # Re-using load_classifier checks the internal global variable _classifier_pipeline
    try:
        loaded = load_classifier()
        cls_status = "Operational" if loaded else "Unavailable/Load Failed"
    except Exception as e: cls_status = f"Error Checking: {type(e).__name__}"

    return StatusResponse(rag_index_status=rag_status, kg_status=kg_status, classifier_status=cls_status)


@app.post("/analyze",
          response_model=Union[FactualAnalysisResponse, MisinformationAnalysisResponse, UrlAnalysisResponse],
          tags=["Analysis"],
          responses={
              400: {"description": "Bad Request", "model": ErrorResponse}, 401: {"description": "Unauthorized", "model": ErrorResponse},
              429: {"description": "Rate Limit Exceeded", "model": ErrorResponse}, 500: {"description": "Internal Server Error", "model": ErrorResponse},
              503: {"description": "Service Unavailable", "model": ErrorResponse},
          })
@cache(expire=CACHE_TIMEOUT) # Apply caching decorator
async def analyze_text(
    request: AnalyzeRequest,
    api_key_dependency: Optional[str] = Depends(get_api_key)
):
    """
    Analyzes input text for misinformation, factual queries, or URL safety.
    Triggers web search synthesis for a consolidated answer if initial confidence is low.
    Requires X-API-Key header if API key security is enabled.
    """
    request_id = str(uuid.uuid4())
    # Set request ID on the current task for potential use in deeper functions (experimental)
    try: asyncio.current_task().request_id = request_id
    except Exception: pass # Ignore if cannot set attribute

    start_time = time.perf_counter()
    input_text = request.text.strip() if request.text else ""

    result_model: Union[FactualAnalysisResponse, MisinformationAnalysisResponse, UrlAnalysisResponse]

    if not input_text:
        logger.warning(f"[ReqID: {request_id}] Received request with empty input text.")
        raise HTTPException(status_code=400, detail={"request_id": request_id, "error": "Bad Request", "message": "Input text cannot be empty."})

    logger.info(f"[ReqID: {request_id}] Received analysis request for: '{input_text[:100]}...'")

    try:
        # 1. Classify Intent
        intent, intent_confidence = classify_intent(input_text)
        logger.info(f"[ReqID: {request_id}] Classified intent as '{intent}' with confidence {intent_confidence:.3f}")

        # --- Routing Logic ---
        CLASSIFICATION_THRESHOLD = 0.60 # Move to config if needed
        effective_intent = intent
        if intent_confidence < CLASSIFICATION_THRESHOLD:
             logger.warning(f"[ReqID: {request_id}] Intent confidence ({intent_confidence:.3f}) below threshold ({CLASSIFICATION_THRESHOLD}). Defaulting from '{intent}' to 'misinfo'.")
             effective_intent = "misinfo" # Fallback

        # --- Route based on effective intent ---
        if effective_intent == "url": result_model = await handle_url_analysis(request_id, input_text)
        elif effective_intent == "misinfo": result_model = await handle_misinfo_analysis(request_id, input_text)
        elif effective_intent == "factual": result_model = await handle_factual_analysis(request_id, input_text)
        else: # Should not happen if classifier labels are url/misinfo/factual
             logger.error(f"[ReqID: {request_id}] Classifier returned unknown intent '{intent}'. Treating as misinfo.")
             result_model = await handle_misinfo_analysis(request_id, input_text)
             if isinstance(result_model, MisinformationAnalysisResponse):
                 result_model.explanation = f"(Intent classification uncertain, processed as misinfo) {result_model.explanation or ''}"

        end_time = time.perf_counter(); processing_time = round((end_time - start_time) * 1000, 2)
        # Assign common fields just before returning
        result_model.processing_time_ms = processing_time; result_model.request_id = request_id; result_model.input_text = input_text
        logger.info(f"[ReqID: {request_id}] Analysis completed in {processing_time:.2f} ms. Final Assessment: {result_model.assessment}, Confidence: {result_model.confidence_score:.3f}, Source: {getattr(result_model, 'data_source', getattr(result_model, 'assessment', 'N/A'))}") # Show assessment if data_source N/A (e.g. URL)
        return result_model

    # --- Exception Handling ---
    except (RateLimitException, ApiException, GroqApiException) as api_exc:
         status_code = 429 if isinstance(api_exc, RateLimitException) else 503
         error_type = "Rate Limit Exceeded" if status_code == 429 else "Service Unavailable"
         log_level = logging.WARNING if status_code == 429 else logging.ERROR
         logger.log(log_level, f"[ReqID: {request_id}] API Error during analysis: {api_exc}", exc_info=True if status_code != 429 else False)
         raise HTTPException(status_code=status_code, detail={"request_id": request_id, "error": error_type, "message": str(api_exc)})
    except HTTPException as he:
        detail_content = he.detail; error_payload = {"request_id": request_id, "error": "API Error", "message": "An error occurred."}
        if isinstance(detail_content, dict): error_payload.update(detail_content); error_payload["request_id"] = request_id # Ensure ID
        elif isinstance(detail_content, str): error_payload["message"] = detail_content
        logger.warning(f"[ReqID: {request_id}] HTTP Exception occurred: Status {he.status_code}, Detail: {error_payload}")
        raise HTTPException(status_code=he.status_code, detail=error_payload) from he
    except Exception as e:
        logger.error(f"[ReqID: {request_id}] Unexpected internal server error during analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"request_id": request_id, "error": "Internal Server Error", "message": f"An unexpected error occurred: {type(e).__name__}"})


# --- Handler Functions for Each Intent ---

async def handle_url_analysis(request_id: str, input_text: str) -> UrlAnalysisResponse:
    """Handles the URL analysis workflow."""
    logger.info(f"[ReqID: {request_id}] Starting URL analysis workflow.")

    # 1. Extract URL and Context
    extracted_url: Optional[str] = None; context_text: Optional[str] = input_text; url_found_info = {}
    url_pattern_scheme = re.compile( r'\b(https?://(?:(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,12}|localhost|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?::\d+)?(?:[/?#]\S*)?)\b', re.IGNORECASE)
    url_pattern_noscheme = re.compile( r'(?:\s|^)((?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,12})\b', re.IGNORECASE)

    scheme_match = url_pattern_scheme.search(input_text)
    if scheme_match:
        found = scheme_match.group(1)
        if len(found) < 500: url_found_info = {"match_str": found, "original": found}; logger.debug(f"[ReqID: {request_id}] Found URL with scheme: {found}")
    else:
        noscheme_match = url_pattern_noscheme.search(input_text)
        if noscheme_match:
            potential_domain_part = noscheme_match.group(1).strip('.').strip('/'); common_tlds = ['.com', '.org', '.net', '.gov', '.edu', '.io', '.co', '.ai', '.uk', '.de', '.ca', '.au']
            if '.' in potential_domain_part and len(potential_domain_part) > 3 and any(potential_domain_part.lower().endswith(tld) for tld in common_tlds):
                potential_url = "https://" + potential_domain_part
                if is_valid_url(potential_url): url_found_info = {"match_str": potential_url, "original": potential_domain_part}; logger.debug(f"[ReqID: {request_id}] Found potential URL (no scheme), assumed https: {potential_url}")

    sanitized_url: Optional[str] = None
    if url_found_info:
        try:
            sanitized_url = sanitize_url_for_scan(url_found_info["match_str"])
            if sanitized_url and is_valid_url(sanitized_url):
                extracted_url = sanitized_url
                context_text = input_text.replace(url_found_info["original"], "[URL]", 1).strip()
                logger.info(f"[ReqID: {request_id}] URL Extracted & Sanitized: {extracted_url} (Original: {url_found_info['original']})")
                if context_text == "[URL]" or not context_text: context_text = None
            else: logger.warning(f"[ReqID: {request_id}] URL found ('{url_found_info['match_str']}') but failed sanitization/validation."); extracted_url = None
        except Exception as e: logger.error(f"[ReqID: {request_id}] Error during URL extraction/sanitization: {e}", exc_info=True); extracted_url = None

    # --- Proceed with Analysis ---
    consolidated_result: Dict[str, Any]
    if extracted_url:
        text_assessment_obj: Optional[TextContextAssessment] = None
        if context_text: text_assessment_obj = _analyze_text_context(request_id, context_text)
        else: text_assessment_obj = TextContextAssessment(suspicion_level="N/A", key_indicators=[]); logger.debug(f"[ReqID: {request_id}] No surrounding text context for URL.")

        try: scan_outputs_dict = await _perform_url_scans(request_id, extracted_url)
        except ValueError as ve: # Catch invalid URL from _perform_url_scans
             logger.error(f"[ReqID: {request_id}] URL scanning failed due to invalid URL input: {ve}")
             consolidated_result = consolidate_url_assessment(input_text, None, None, {}) # Pass empty scans
             consolidated_result.update({"assessment": "Analysis Failed", "analysis_summary": f"URL processing error: {ve}", "confidence_score": 0.1, "evidence_notes": [f"URL processing error: {ve}"]})
             # Return early with error structure
             # Processing time etc are set by the main endpoint wrapper
             return UrlAnalysisResponse(request_id="dummy", input_text="dummy", processing_time_ms=0.0, **consolidated_result)
        except (ApiException, RateLimitException) as scan_api_err:
            logger.error(f"[ReqID: {request_id}] Critical failure during URL scanning: {scan_api_err}")
            consolidated_result = consolidate_url_assessment(input_text, extracted_url, text_assessment_obj, {}) # Pass empty scans
            consolidated_result.update({"assessment": "Analysis Failed", "analysis_summary": f"Failed to get results from scanners: {scan_api_err}", "confidence_score": 0.1, "evidence_notes": [f"Scanning services unavailable: {scan_api_err}"]})
            return UrlAnalysisResponse(request_id="dummy", input_text="dummy", processing_time_ms=0.0, **consolidated_result)

        # Consolidate if scans ran (even if some failed individually)
        consolidated_result = consolidate_url_assessment(input_text, extracted_url, text_assessment_obj, scan_outputs_dict)
    else: # No processable URL found
        logger.warning(f"[ReqID: {request_id}] No processable URL found in input text for analysis.")
        consolidated_result = consolidate_url_assessment(input_text, None, None, {}) # Use logic for N/A URL

    # Assemble and return the final response
    response = UrlAnalysisResponse(request_id="dummy", input_text="dummy", processing_time_ms=0.0, **consolidated_result)
    logger.info(f"[ReqID: {request_id}] URL Analysis Completed. Assessment: {response.assessment}, Confidence: {response.confidence_score:.3f}")
    return response


async def handle_misinfo_analysis(request_id: str, input_text: str) -> MisinformationAnalysisResponse:
    """Handles Misinfo analysis: RAG -> Direct LLM -> Web Search Synthesis Fallback."""
    logger.info(f"[ReqID: {request_id}] Starting Misinformation analysis workflow (with web fallback).")
    global rag_processor
    if not rag_processor:
        logger.error(f"[ReqID: {request_id}] RAG processor unavailable. Skipping RAG stage.")
        # Continue without RAG

    # Initialize response fields
    assessment: Literal["Likely Factual", "Likely Misleading", "Opinion", "Needs Verification / Uncertain", "Contradictory Information Found"] = "Needs Verification / Uncertain"
    confidence = 0.0; explanation = "Analysis pending."; evidence: List[EvidenceItem] = []
    kg_insights: Optional[str] = None; data_source: Literal["RAG", "LLM Internal Knowledge", "Web Search", "Web Search Synthesis"] = "LLM Internal Knowledge"
    key_issues: List[str] = []; verifiable_claims: List[str] = []; raw_llm_output: Optional[str] = None

    # KG Query
    try: extracted_ents = extract_entities(input_text); kg_insights = query_kg_for_entities(extracted_ents) if extracted_ents else None
    except Exception as kg_err: logger.error(f"[ReqID: {request_id}] KG query error: {kg_err}", exc_info=True); kg_insights = "Knowledge Graph query failed."

    # --- Stage 1: Try RAG ---
    rag_sufficient = False
    if rag_processor: # Only attempt RAG if processor is available
        try:
            logger.debug(f"[ReqID: {request_id}] Attempting RAG query for misinfo check.")
            rag_response, rag_sources = await rag_processor.query_rag(input_text, use_for="misinfo_check")
            if rag_response and rag_sources:
                rag_sufficient = True; data_source = "RAG"; explanation = rag_response
                evidence = [EvidenceItem(source=str(s.get('source','RAG Document')), snippet=str(s.get('snippet', ''))[:300]+"...", assessment_note="Retrieved via RAG") for s in rag_sources]
                assessment, confidence = _assess_llm_response_keywords(explanation, base_confidence=0.75) # Use helper, higher base for RAG
                logger.info(f"[ReqID: {request_id}] Initial check using RAG result. Assessment: {assessment}, Conf: {confidence:.2f}")
            else: logger.info(f"[ReqID: {request_id}] RAG query insufficient. Proceeding."); explanation = "RAG system did not find sufficient context."
        except Exception as rag_err: logger.error(f"[ReqID: {request_id}] Error during RAG query stage: {rag_err}", exc_info=True); explanation = f"RAG query failed: {rag_err}"
    else: explanation = "RAG system unavailable."

    # --- Stage 2: Try Direct LLM (if RAG failed/insufficient) ---
    if not rag_sufficient:
        try:
            logger.info(f"[ReqID: {request_id}] Attempting direct LLM analysis via Groq for misinfo.")
            direct_groq_result = await analyze_misinformation_groq(input_text)
            if direct_groq_result and direct_groq_result.get("category") != "error":
                data_source = "LLM Internal Knowledge"; cat = direct_groq_result.get("category", "other")
                map_assessment = {"likely_factual": "Likely Factual", "likely_misleading": "Likely Misleading", "opinion": "Opinion", "satire": "Opinion", "needs_verification": "Needs Verification / Uncertain", "contradictory": "Contradictory Information Found", "other": "Needs Verification / Uncertain" }
                assessment = map_assessment.get(cat, "Needs Verification / Uncertain"); confidence = direct_groq_result.get("confidence", 0.4)
                explanation = direct_groq_result.get("explanation", "No explanation provided by LLM."); key_issues = direct_groq_result.get("key_issues", [])
                verifiable_claims = direct_groq_result.get("verifiable_claims", [])
                evidence = [EvidenceItem(source="LLM Direct Analysis", snippet=explanation[:200]+"...", assessment_note=f"LLM raw category: {cat}")]
                raw_llm_output = direct_groq_result.get("raw_response")
                logger.info(f"[ReqID: {request_id}] Initial check using direct LLM. Assessment: {assessment}, Conf: {confidence:.2f}")
            else:
                logger.warning(f"[ReqID: {request_id}] Direct LLM analysis failed or returned error structure. Result: {direct_groq_result}")
                explanation = explanation + " | Direct LLM analysis failed." if explanation != "Analysis pending." else "Direct LLM analysis failed."
                assessment = "Needs Verification / Uncertain"; confidence = 0.1
        except (RateLimitException, GroqApiException, ApiException) as direct_llm_err:
            logger.error(f"[ReqID: {request_id}] API error during direct LLM analysis stage: {direct_llm_err}", exc_info=False)
            explanation = explanation + f" | Direct LLM failed: {direct_llm_err}" if explanation != "Analysis pending." else f"Direct LLM failed: {direct_llm_err}"
            assessment = "Needs Verification / Uncertain"; confidence = 0.1
        except Exception as e:
             logger.error(f"[ReqID: {request_id}] Unexpected error during direct LLM analysis stage: {e}", exc_info=True)
             explanation = explanation + f" | Unexpected error in direct LLM: {e}" if explanation != "Analysis pending." else f"Unexpected error in direct LLM: {e}"
             assessment = "Needs Verification / Uncertain"; confidence = 0.1

    # --- Stage 3: Web Search Synthesis Fallback (Trigger Check) ---
    trigger_fallback = False
    uncertain_assessments = {"Needs Verification / Uncertain", "Contradictory Information Found"}
    if assessment in uncertain_assessments or confidence < WEB_FALLBACK_THRESHOLD:
        trigger_fallback = True; logger.info(f"[ReqID: {request_id}] Triggering web search synthesis fallback. Reason: Assessment='{assessment}', Confidence={confidence:.2f}")

    if trigger_fallback:
        try:
            logger.info(f"[ReqID: {request_id}] Executing web search for synthesis fallback.")
            search_results = await perform_search(input_text)
            if search_results:
                synthesis_result = await _synthesize_from_web_results(request_id, input_text, search_results, "misinfo")
                if synthesis_result.get("data_source") == "Web Search Synthesis":
                    # Overwrite initial assessment with synthesized results
                    assessment = synthesis_result.get("assessment", assessment); confidence = synthesis_result.get("confidence_score", confidence)
                    explanation = synthesis_result.get("explanation", explanation) # This now contains the detailed synthesized explanation
                    evidence = synthesis_result.get("evidence", evidence) # Evidence now lists web sources used
                    data_source = "Web Search Synthesis"; key_issues = []; verifiable_claims = [] # Reset these as they pertain to synthesis
                    raw_llm_output = synthesis_result.get("raw_llm_output")
                    logger.info(f"[ReqID: {request_id}] Updated response via web synthesis. New Assessment: {assessment}, New Conf: {confidence:.2f}")
                else: logger.warning(f"[ReqID: {request_id}] Web synthesis helper failed. Reason: {synthesis_result.get('explanation', 'Unknown')}"); explanation += f" | Synthesis process error: {synthesis_result.get('explanation', 'Unknown')}"; data_source = synthesis_result.get("data_source", "Web Search")
            else: logger.warning(f"[ReqID: {request_id}] Web search fallback found no results."); explanation += " | Web search attempted but found no results."; data_source = "Web Search"
        except (RateLimitException, ApiException) as search_api_err: logger.error(f"[ReqID: {request_id}] API error during web search fallback: {search_api_err}"); explanation += f" | Web search fallback failed (API error)"; data_source = "Web Search"
        except Exception as search_err: logger.error(f"[ReqID: {request_id}] Unexpected error during web search fallback: {search_err}", exc_info=True); explanation += f" | Web search fallback failed (unexpected error)"; data_source = "Web Search"

    # --- Update KG ---
    try:
        if extracted_ents: add_claim_to_graph(input_text, str(assessment), f"user_input:{request_id}", extracted_ents); logger.info(f"[ReqID: {request_id}] Updated KG with final assessment: {assessment}")
    except Exception as e: logger.error(f"[ReqID: {request_id}] KG update failed: {e}", exc_info=True)

    # --- Construct Final Response ---
    return MisinformationAnalysisResponse(
        request_id="dummy", input_text="dummy", processing_time_ms=0.0, # Placeholders
        assessment=assessment, confidence_score=round(confidence, 3), explanation=explanation or "Analysis yielded no explanation.",
        evidence=evidence, data_source=data_source, key_issues_identified=key_issues, verifiable_claims=verifiable_claims,
        knowledge_graph_insights=kg_insights )


async def handle_factual_analysis(request_id: str, input_text: str) -> FactualAnalysisResponse:
    """Handles Factual QA: RAG -> Direct LLM -> Web Search Synthesis Fallback."""
    logger.info(f"[ReqID: {request_id}] Starting Factual analysis workflow (with web fallback).")
    global rag_processor
    if not rag_processor: logger.error(f"[ReqID: {request_id}] RAG processor unavailable. Skipping RAG stage.")

    # Initialize
    assessment: Literal["Likely Factual", "Opinion", "Needs Verification / Uncertain", "Contradictory Information Found"] = "Needs Verification / Uncertain"
    confidence = 0.0; answer = "Analysis pending."; explanation: Optional[str] = None; evidence: List[EvidenceItem] = []
    kg_insights: Optional[str] = None; data_source: Literal["RAG", "LLM Internal Knowledge", "Web Search", "Web Search Synthesis", "N/A"] = "LLM Internal Knowledge"
    raw_llm_output: Optional[str] = None

    # KG Query
    try: extracted_ents = extract_entities(input_text); kg_insights = query_kg_for_entities(extracted_ents) if extracted_ents else None
    except Exception as kg_err: logger.error(f"[ReqID: {request_id}] KG query error: {kg_err}", exc_info=True); kg_insights = "Knowledge Graph query failed."

    # --- Stage 1: Try RAG ---
    rag_sufficient = False
    if rag_processor:
        try:
            logger.debug(f"[ReqID: {request_id}] Attempting RAG query for factual QA.")
            rag_response, rag_sources = await rag_processor.query_rag(input_text, use_for="factual_qa")
            if rag_response and rag_sources:
                rag_sufficient = True; data_source="RAG"; answer=rag_response
                evidence=[EvidenceItem(source=str(s.get('source','RAG Document')), snippet=str(s.get('snippet', ''))[:300]+"...", assessment_note="Retrieved via RAG") for s in rag_sources]
                l_ans = answer.lower(); rag_uncertainty_markers = ["don't know", "context doesn't provide", "cannot answer", "unable to provide", "not specified in the context", "insufficient information"]
                if any(marker in l_ans for marker in rag_uncertainty_markers): assessment="Needs Verification / Uncertain"; confidence=0.4; explanation = "RAG context didn't contain answer."
                elif "opinion" in l_ans and len(answer)<250: assessment="Opinion"; confidence=0.7; explanation = "Context seems opinion-based."
                else: assessment="Likely Factual"; confidence=0.88
                logger.info(f"[ReqID: {request_id}] Initial check using RAG. Assessment: {assessment}, Conf: {confidence:.2f}")
            else: logger.info(f"[ReqID: {request_id}] RAG query insufficient."); answer = "RAG system did not find sufficient context."
        except Exception as rag_err: logger.error(f"[ReqID: {request_id}] Error during RAG query stage: {rag_err}", exc_info=True); answer = f"RAG query failed: {rag_err}"
    else: answer = "RAG system unavailable."

    # --- Stage 2: Try Direct LLM (if RAG failed/insufficient) ---
    if not rag_sufficient:
        try:
            logger.info(f"[ReqID: {request_id}] Attempting direct LLM factual Q&A via Groq.")
            direct_groq_result = await ask_groq_factual(input_text)
            if direct_groq_result and direct_groq_result.get("confidence_level") != "error" and direct_groq_result.get("answer"):
                data_source = "LLM Internal Knowledge"; answer = direct_groq_result.get("answer", "").strip(); g_conf = direct_groq_result.get("confidence_level", "low")
                evidence=[EvidenceItem(source="LLM Direct Analysis", snippet=answer[:200]+"...", assessment_note=f"LLM self-assessed confidence: {g_conf}")]
                raw_llm_output = answer; map_conf={"high": 0.85, "medium": 0.65, "low": 0.35}; confidence=map_conf.get(g_conf, 0.3)
                llm_uncertainty_markers = ["don't know", "cannot answer", "not sure", "unable to provide", "no information", "limited knowledge", "beyond my cutoff"]
                if any(marker in answer.lower() for marker in llm_uncertainty_markers): assessment="Needs Verification / Uncertain"; confidence = max(0.1, confidence * 0.7)
                elif g_conf == "low": assessment="Needs Verification / Uncertain"
                else: assessment="Likely Factual"
                logger.info(f"[ReqID: {request_id}] Initial check using direct LLM. Assessment: {assessment}, Conf: {confidence:.2f}")
            else:
                logger.warning(f"[ReqID: {request_id}] Direct LLM factual Q&A failed/no answer. Result: {direct_groq_result}")
                answer = answer + " | Direct LLM failed." if answer != "Analysis pending." else "Direct LLM failed."
                assessment = "Needs Verification / Uncertain"; confidence = 0.1
        except (RateLimitException, GroqApiException, ApiException) as direct_llm_err:
            logger.error(f"[ReqID: {request_id}] API error during direct LLM stage: {direct_llm_err}")
            answer = answer + f" | Direct LLM failed: {direct_llm_err}" if answer != "Analysis pending." else f"Direct LLM failed: {direct_llm_err}"
            assessment = "Needs Verification / Uncertain"; confidence = 0.1
        except Exception as e:
             logger.error(f"[ReqID: {request_id}] Unexpected error during direct LLM stage: {e}", exc_info=True)
             answer = answer + f" | Unexpected error in direct LLM: {e}" if answer != "Analysis pending." else f"Unexpected error in direct LLM: {e}"
             assessment = "Needs Verification / Uncertain"; confidence = 0.1

    # --- Stage 3: Web Search Synthesis Fallback (Trigger Check) ---
    trigger_fallback = False; uncertain_assessments = {"Needs Verification / Uncertain"}
    current_answer_indicates_uncertainty = False
    if isinstance(answer, str):
        uncertainty_phrases = ["don't know", "cannot answer", "not sure", "unable to provide", "no information", "failed", "insufficient context", "did not find"]
        current_answer_indicates_uncertainty = any(phrase in answer.lower() for phrase in uncertainty_phrases)
    if assessment in uncertain_assessments or confidence < WEB_FALLBACK_THRESHOLD or current_answer_indicates_uncertainty:
        trigger_fallback = True; logger.info(f"[ReqID: {request_id}] Triggering web search synthesis fallback. Reason: Assessment='{assessment}', Conf={confidence:.2f}, AnsUncertain='{current_answer_indicates_uncertainty}'")

    if trigger_fallback:
        try:
            logger.info(f"[ReqID: {request_id}] Executing web search for synthesis fallback.")
            search_results = await perform_search(input_text)
            if search_results:
                synthesis_result = await _synthesize_from_web_results(request_id, input_text, search_results, "factual")
                if synthesis_result.get("data_source") == "Web Search Synthesis":
                    # Overwrite initial assessment/answer with synthesized results
                    assessment = synthesis_result.get("assessment", assessment); confidence = synthesis_result.get("confidence_score", confidence)
                    answer = synthesis_result.get("answer", answer) # This is the synthesized narrative answer
                    explanation = synthesis_result.get("explanation", explanation) # Could add note here if needed
                    evidence = synthesis_result.get("evidence", evidence); data_source = "Web Search Synthesis"
                    raw_llm_output = synthesis_result.get("raw_llm_output")
                    logger.info(f"[ReqID: {request_id}] Updated response via web synthesis. New Assessment: {assessment}, New Conf: {confidence:.2f}")
                else: logger.warning(f"[ReqID: {request_id}] Web synthesis helper failed. Reason: {synthesis_result.get('explanation', 'Unknown')}"); answer += f" | Synthesis process error."; data_source = synthesis_result.get("data_source", "Web Search")
            else: logger.warning(f"[ReqID: {request_id}] Web search fallback found no results."); answer += " | Web search attempted but found no results."; data_source = "Web Search"
        except (RateLimitException, ApiException) as search_api_err: logger.error(f"[ReqID: {request_id}] API error during web search fallback: {search_api_err}"); answer += f" | Web search fallback failed (API error)."; data_source = "Web Search"
        except Exception as search_err: logger.error(f"[ReqID: {request_id}] Unexpected error during web search fallback: {search_err}", exc_info=True); answer += f" | Web search fallback failed (unexpected error)."; data_source = "Web Search"

    # --- Construct Final Response ---
    return FactualAnalysisResponse(
        request_id="dummy", input_text="dummy", processing_time_ms=0.0, # Placeholders
        assessment=assessment, answer=answer or "Analysis could not provide an answer.", explanation=explanation,
        confidence_score=round(confidence, 3), data_source=data_source, supporting_evidence=evidence,
        knowledge_graph_insights=kg_insights )

# --- Optional: Uvicorn Runner ---
# if __name__ == "__main__":
#     import uvicorn
#     # ... (Uvicorn running code from previous version) ...
# --- Optional: Run with Uvicorn for local testing ---
# Note: Typically run using a process manager like gunicorn/uvicorn directly in production
# Add this to your FastAPI startup script
import socket
import netifaces
from typing import List

def get_network_info():
    """Returns (local_ip, all_ips)"""
    try:
        # Method 1: Best for getting routable IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        
        # Method 2: List all IPs
        all_ips = []
        for interface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(interface).get(netifaces.AF_INET, [])
            for addr in addrs:
                if addr['addr'] != local_ip:
                    all_ips.append(addr['addr'])
        
        return local_ip, all_ips
    except Exception as e:
        print(f"Could not determine IP: {e}")
        return "127.0.0.1", []

if __name__ == "__main__":
    local_ip, all_ips = get_network_info()
    print(f"Starting server on:\n- Local: http://127.0.0.1:8000\n- Network: http://{local_ip}:8000")
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",  # Important! Bind to all interfaces
        port=8000,
        reload=True
    )