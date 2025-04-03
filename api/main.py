# api/main.py

import logging
import time
import asyncio
import uuid
import os
import re  # <--- IMPORT REQUIRED
from contextlib import asynccontextmanager
from typing import Dict, Optional, List, Literal, Any, Union , Tuple# <-- Union imported

import httpx # Keep httpx for potential use, though less needed here directly
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
    from .models import (AnalyzeRequest, BaseAnalysisResponse, FactualAnalysisResponse, MisinformationAnalysisResponse,
                        UrlAnalysisResponse, StatusResponse, ErrorResponse, TextContextAssessment, ScanResultDetail,
                        UrlScanResults, EvidenceItem)
    from .classifier import classify_intent, load_classifier
    from .groq_utils import ( # Ensure all used functions are imported
        query_groq, setup_groq_client, close_groq_client, GroqApiException,
        ask_groq_factual, analyze_misinformation_groq
    )
    from .langchain_utils import RealTimeDataProcessor # Handles RAG + Cohere
    from .utils import get_config, setup_logging, is_valid_url, RateLimitException, ApiException, sanitize_url_for_scan
    from .vt_utils import check_virustotal, parse_vt_result
    from .ipqs_utils import check_ipqs, parse_ipqs_result
    from .urlscan_utils import check_urlscan_existing_results, parse_urlscan_result
    from .search_api_utils import perform_search
    from .kg_utils import load_graph, save_graph, load_spacy_model, extract_entities, add_claim_to_graph, query_kg_for_entities
except ImportError as e:
     print(f"ERROR: Failed to import necessary modules: {e}")
     print("Ensure all util files exist and Python can find the 'api' package.")
     exit(1)


# --- Basic Setup ---
setup_logging()
logger = logging.getLogger(__name__)
load_dotenv()
CONFIG = get_config()

# --- Globals (initialized in lifespan) ---
rag_processor: Optional[RealTimeDataProcessor] = None
redis_client: Optional[aioredis.Redis] = None # Hold redis client ref for shutdown
shutdown_event = asyncio.Event()

# --- API Key Security ---
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False) # auto_error=False allows handling optional key
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")
API_KEY_ENABLED = CONFIG.get("security", {}).get("enable_api_key_auth", False) # Defaulting to False based on user need

async def get_api_key(key: str = Security(api_key_header)):
    """Dependency to validate the API Key if security is enabled."""
    if not API_KEY_ENABLED: # If security is disabled in config, allow request
        logger.debug("API Key auth disabled.")
        return None
    if not key:
         logger.info("Request rejected: API Key required via X-API-Key header.")
         raise HTTPException(status_code=401, detail="API Key required via X-API-Key header.")
    if not INTERNAL_API_KEY:
         logger.error("API Key security is enabled, but INTERNAL_API_KEY is not set in .env. Denying request.")
         raise HTTPException(status_code=500, detail="Internal server configuration error: API Key auth misconfigured")
    if key == INTERNAL_API_KEY:
        logger.debug("API Key validated successfully.")
        return key
    else:
        logger.warning("Request rejected: Invalid API Key provided.")
        raise HTTPException(status_code=401, detail="Invalid API Key.")

# --- App Lifespan Management (Model Loading, Cache Init) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_processor, redis_client # Add redis_client to global
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
        redis_client = None

    # 3. Load Classifier
    logger.info("Loading intent classifier model...")
    if not load_classifier(): # Tries to load the model into the pipeline
        logger.error("Failed to load intent classifier model. Startup aborted.")
        raise RuntimeError("Failed to load essential intent classifier model.")
    logger.info("Intent classifier loaded.")

    # 4. Initialize RAG Processor
    logger.info("Initializing RAG processor...")
    try:
        rag_processor = RealTimeDataProcessor()
        if not rag_processor.embeddings or not rag_processor.vector_store:
             logger.warning("RAG processor failed initial setup (embeddings or vector store missing/failed). RAG functionality will be limited.")
        else:
             logger.info("RAG processor initialized successfully.")
    except Exception as e:
         logger.error(f"Critical error initializing RAG processor: {e}", exc_info=True)
         rag_processor = None

    # 5. Load Knowledge Graph & SpaCy
    logger.info("Loading Knowledge Graph...")
    try: load_graph()
    except Exception as e: logger.error(f"Failed to load knowledge graph: {e}", exc_info=True)

    logger.info("Loading SpaCy model for KG NER...")
    try:
        if not load_spacy_model(): logger.warning("Failed to load SpaCy model. KG entity extraction disabled.")
    except Exception as e: logger.error(f"Error loading SpaCy model: {e}", exc_info=True)


    logger.info("Application startup complete.")
    yield  # API is now running

    # --- Shutdown Sequence ---
    logger.info("Application shutdown initiated...")
    shutdown_event.set()

    try: save_graph()
    except Exception as e: logger.error(f"Error saving graph on shutdown: {e}")
    await close_groq_client()
    if redis_client:
        try: await redis_client.close(); logger.info("Redis connection closed.")
        except Exception as e: logger.error(f"Error closing Redis connection: {e}")

    logger.info("Application shutdown complete.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Hack the Hoax - Misinformation Detector API",
    description="Advanced API using RAG, LLMs, multiple URL scanners, and KG for real-time analysis.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- CORS Configuration ---
allowed_origins = CONFIG.get("api", {}).get("cors_allowed_origins", ["*"])
logger.info(f"Configuring CORS for origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- Consolidation Logic for URL Analysis ---
def consolidate_url_assessment(
    url: str,
    text_assessment: Optional[TextContextAssessment],
    scan_outputs: Dict[str, Optional[ScanResultDetail]]
) -> Dict[str, Any]:
    """Consolidates text and scanner results into a final URL assessment dict."""
    evidence_notes = []; confidence = 0.5
    AssessmentType = Literal["Malicious", "Phishing", "Spam", "Suspicious", "Likely Safe", "Uncertain", "Analysis Failed"]
    assessment: AssessmentType = "Uncertain"; analysis_summary = "Analysis initiated."
    malicious_score = 0.0; spam_score = 0.0; suspicion_score = 0.0; safety_score = 0.0

    # Text Context Processing
    text_suspicion = "N/A"
    if text_assessment:
        text_suspicion = text_assessment.suspicion_level
        if text_suspicion == "High": suspicion_score += 3.0; spam_score += 1.0; evidence_notes.append("Text context flagged as highly suspicious.")
        elif text_suspicion == "Medium": suspicion_score += 1.5; evidence_notes.append("Text context flagged as medium suspicion.")
        if text_assessment.key_indicators: evidence_notes.extend([f"Text Indicator: {ind}" for ind in text_assessment.key_indicators])

    # Scanner Results Processing
    vt_res = scan_outputs.get('virustotal')
    ipqs_res = scan_outputs.get('ipqualityscore')
    urlscan_res = scan_outputs.get('urlscanio')
    successful_scans = 0
    scan_details = {"virustotal": vt_res, "ipqualityscore": ipqs_res, "urlscanio": urlscan_res}

    if vt_res and vt_res.status == 'success' and vt_res.details:
        successful_scans += 1; d = vt_res.details; pos = d.get('positives', 0); total = d.get('total_engines', 0); assess = d.get('assessment', 'N/A')
        evidence_notes.append(f"VT: {pos}/{total} engines ({assess}).")
        if assess == 'malicious': malicious_score += 3.0 + pos * 0.05; suspicion_score += 2.0
        elif assess == 'suspicious': suspicion_score += 1.0 + pos * 0.1
        elif assess == 'likely_safe': safety_score += 2.0
    elif vt_res and vt_res.status != 'success': evidence_notes.append(f"VT: Status {vt_res.status} ({vt_res.details.get('message', 'No details') if vt_res.details else 'N/A'}).")

    if ipqs_res and ipqs_res.status == 'success' and ipqs_res.details:
        successful_scans += 1; d = ipqs_res.details; risk = d.get('risk_score', 0); threat = d.get('threat_category', 'N/A'); is_p = d.get('is_phishing'); is_m = d.get('is_malware'); is_s = d.get('is_spam'); age = str(d.get('domain_age_days', 'N/A'))
        evidence_notes.append(f"IPQS: Risk {risk}, Cat '{threat}', Phish:{is_p}, Mal:{is_m}.")
        if is_p: malicious_score += 6.0; suspicion_score += 2.0
        if is_m: malicious_score += 5.0; suspicion_score += 1.0
        if is_s: spam_score += 3.0
        if d.get('assessment_category') == 'high_risk': suspicion_score += 4.0
        elif d.get('assessment_category') == 'medium_risk': suspicion_score += 2.0
        elif d.get('assessment_category') == 'low_risk': safety_score += 1.0
        is_young = age.isdigit() and int(age) < 90 or ('day' in age.lower() and 'year' not in age.lower())
        if is_young: suspicion_score += 1.0; evidence_notes.append("IPQS: Domain age appears recent.")
    elif ipqs_res and ipqs_res.status != 'success': evidence_notes.append(f"IPQS: Status {ipqs_res.status} ({ipqs_res.details.get('message', 'No details') if ipqs_res.details else 'N/A'}).")

    if urlscan_res and urlscan_res.status == 'scan_found' and urlscan_res.details:
        successful_scans += 1; d = urlscan_res.details; tags = d.get('tags', []); score = d.get('score',0)
        tags_str = ', '.join(tags) if tags else 'None'; evidence_notes.append(f"URLScan: Found scan ({d.get('scan_date', 'N/A')[:10]}), Score:{score}. Malicious:{d.get('verdict_malicious')}. Tags:[{tags_str[:50]}...]")
        if d.get('verdict_malicious'): malicious_score += 5.0; suspicion_score += 2.0
        elif d.get('assessment_category') == 'suspicious': suspicion_score += 3.0
        if any(t in tags for t in ['phishing', 'malware', 'suspicious', 'crypto-scam', 'hacked', 'dynamic dns']): suspicion_score += 2.5

        if score > 75: # Adjusted threshold for more significance
             suspicion_score += 1.5
        elif score < 10: # Score is very low (good)
             safety_score += 0.5

    elif urlscan_res and urlscan_res.status != 'scan_found': # Combined logging for rate limit/error/no_scan
         evidence_notes.append(f"URLScan: Status {urlscan_res.status} ({urlscan_res.details.get('message', 'No details') if urlscan_res.details else 'N/A'}).")

    # Final Assessment Logic
    MALICIOUS_CONFIRMED_THR = 5.0; SUSPICIOUS_THR = 3.5; SPAM_THR = 3.0; SAFE_THR = 1.5
    net_safety_score = safety_score - suspicion_score

    if malicious_score >= MALICIOUS_CONFIRMED_THR:
        is_likely_phishing = (ipqs_res and ipqs_res.details and ipqs_res.details.get('is_phishing')) or \
                             any(t == 'phishing' for t in (urlscan_res.details.get('tags', []) if urlscan_res and urlscan_res.details else []))
        assessment = "Phishing" if is_likely_phishing else "Malicious"
        confidence = min(0.9 + malicious_score * 0.01, 0.99); analysis_summary = f"High risk of {assessment.lower()} detected."
    elif spam_score >= SPAM_THR and text_suspicion in ["High", "Medium"]:
        assessment = "Spam"; confidence = min(0.7 + spam_score * 0.05, 0.90); analysis_summary = "URL potentially related to spam."
    elif suspicion_score >= SUSPICIOUS_THR or (suspicion_score > 1 and text_suspicion == "High"): # Increased sensitivity with high text suspicion
        assessment = "Suspicious"; confidence = min(0.55 + suspicion_score * 0.06, 0.88); analysis_summary = "URL flagged as potentially suspicious."
    elif net_safety_score >= SAFE_THR :
        assessment = "Likely Safe"; confidence = min(0.75 + safety_score * 0.05, 0.95); analysis_summary = "URL appears likely safe based on scans."
    else: # Default to Uncertain
        assessment = "Uncertain"; confidence = max(0.4 - suspicion_score * 0.1, 0.2); analysis_summary = "Analysis inconclusive or requires verification."

    # Handle cases with failed scans
    if successful_scans == 0 and url != "N/A": # Only if a URL existed but all scans failed
        assessment = "Analysis Failed"; confidence = 0.1
        analysis_summary = "Could not retrieve results from external URL scanners."; evidence_notes = ["Failed to get results from scanning services."]
    # Ensure scan_details has placeholders if completely missing (should be handled by initialisation though)
    for k in scan_details:
        if scan_details[k] is None: scan_details[k] = ScanResultDetail(status="error", details={"message": "Scanner unavailable or failed"})

    return {"assessment": assessment, "scanned_url": url, "confidence_score": round(confidence, 3), "analysis_summary": analysis_summary,
            "text_context_assessment": text_assessment, "scan_results": UrlScanResults(**scan_details), "evidence_notes": evidence_notes }


# --- API Endpoints ---

@app.get("/status", response_model=StatusResponse, tags=["General"])
async def get_status():
    """Provides the operational status of the API and its components."""
    rag_status = "Operational" if rag_processor and rag_processor.vector_store else "Degraded/Unavailable"
    kg_status, cls_status = "Unavailable", "Unavailable"
    try: kg = load_graph(); kg_status = "Operational" if kg is not None else "Unavailable"
    except Exception as e: kg_status = f"Error Loading: {e}"
    try: loaded = load_classifier(); cls_status = "Operational" if loaded else "Unavailable"
    except Exception as e: cls_status = f"Error Loading: {e}"
    return StatusResponse(rag_index_status=rag_status, kg_status=kg_status, classifier_status=cls_status)

CACHE_TIMEOUT = CONFIG.get('cache', {}).get('default_ttl_seconds', 300)

@app.post("/analyze",
          # Define the possible successful response models using Union
          response_model=Union[FactualAnalysisResponse, MisinformationAnalysisResponse, UrlAnalysisResponse],
          tags=["Analysis"],
          responses={ # Define error responses separately
              400: {"description": "Bad Request", "model": ErrorResponse}, 401: {"description": "Unauthorized"},
              429: {"description": "Rate Limit Exceeded", "model": ErrorResponse}, 500: {"description": "Internal Server Error", "model": ErrorResponse},
              503: {"description": "Service Unavailable", "model": ErrorResponse},
          })
@cache(expire=CACHE_TIMEOUT)
async def analyze_text(
    request: AnalyzeRequest,
    # Apply API Key dependency conditionally based on config
    api_key_dependency: Optional[str] = Depends(get_api_key) # get_api_key handles the conditional logic
):
    """
    Analyzes input text for misinformation, factual queries, or URL safety.
    Requires X-API-Key header if API key security is enabled in config.yaml.
    """
    request_id = str(uuid.uuid4()); start_time = time.perf_counter(); input_text = request.text.strip()
    result_model: Union[FactualAnalysisResponse, MisinformationAnalysisResponse, UrlAnalysisResponse] # Type hint

    if not input_text:
        logger.warning(f"[ReqID: {request_id}] Received request with empty input text.")
        raise HTTPException(status_code=400, detail={"request_id": request_id, "error": "Bad Request", "message": "Input text cannot be empty."})

    logger.info(f"[ReqID: {request_id}] Received analysis request for: '{input_text[:100]}...'")

    try:
        # 1. Classify Intent
        intent, intent_confidence = classify_intent(input_text)
        logger.info(f"[ReqID: {request_id}] Classified intent as '{intent}' with confidence {intent_confidence:.3f}")

        # --- Routing Logic ---
        # Confidence Threshold for Classification (example)
        CLASSIFICATION_THRESHOLD = 0.60 # Only trust classifications above this score
        effective_intent = intent
        if intent_confidence < CLASSIFICATION_THRESHOLD:
             logger.warning(f"[ReqID: {request_id}] Intent confidence ({intent_confidence:.3f}) below threshold ({CLASSIFICATION_THRESHOLD}). Defaulting to 'misinfo'.")
             effective_intent = "misinfo" # Fallback if confidence is low

        # --- Route based on effective intent ---
        if effective_intent == "url": result_model = await handle_url_analysis(request_id, input_text)
        elif effective_intent == "misinfo": result_model = await handle_misinfo_analysis(request_id, input_text)
        elif effective_intent == "factual": result_model = await handle_factual_analysis(request_id, input_text)
        else: # Should theoretically not happen if labels are url/misinfo/factual
             logger.error(f"[ReqID: {request_id}] Fallback from unknown effective intent '{effective_intent}'. Treating as misinfo.")
             result_model = await handle_misinfo_analysis(request_id, input_text)
             if isinstance(result_model, MisinformationAnalysisResponse): result_model.explanation = f"(Intent unclear, processed as misinfo) {result_model.explanation}"


        end_time = time.perf_counter(); processing_time = round((end_time - start_time) * 1000, 2)
        # Assign common fields just before returning
        result_model.processing_time_ms = processing_time; result_model.request_id = request_id; result_model.input_text = input_text
        logger.info(f"[ReqID: {request_id}] Analysis completed in {processing_time:.2f} ms. Final Assessment: {result_model.assessment}")
        return result_model

    # --- Exception Handling ---
    except (RateLimitException, ApiException, GroqApiException) as api_exc:
         status_code = 429 if isinstance(api_exc, RateLimitException) else 503
         error_type = "Rate Limit Exceeded" if status_code == 429 else "Service Unavailable"
         log_level = logging.WARNING if status_code == 429 else logging.ERROR
         logger.log(log_level, f"[ReqID: {request_id}] API Error during analysis: {api_exc}", exc_info=True if status_code != 429 else False)
         raise HTTPException(status_code=status_code, detail={"request_id": request_id, "error": error_type, "message": str(api_exc)})
    except HTTPException as he:
        detail = he.detail; detail_dict = {"message": detail} if isinstance(detail, str) else (detail if isinstance(detail,dict) else {})
        if "request_id" not in detail_dict: detail_dict["request_id"] = request_id
        logger.warning(f"[ReqID: {request_id}] HTTP Exception occurred: Status {he.status_code}, Detail: {detail_dict}")
        raise HTTPException(status_code=he.status_code, detail=detail_dict) from he
    except Exception as e:
        logger.error(f"[ReqID: {request_id}] Unexpected internal server error during analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"request_id": request_id, "error": "Internal Server Error", "message": "An unexpected error occurred."})


# --- Handler Functions for Each Intent ---

async def handle_url_analysis(request_id: str, input_text: str) -> UrlAnalysisResponse:
    """Handles the URL analysis workflow."""
    logger.info(f"[ReqID: {request_id}] Starting URL analysis workflow.")

    # 1. Extract URL and Context (Corrected v3)
    url: Optional[str] = None; context_text: Optional[str] = input_text; url_found_info = {}
    url_pattern_scheme = re.compile(r'https?://(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|localhost|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?::\d+)?(?:/?|[/?]\S+)', re.IGNORECASE)
    url_pattern_noscheme = re.compile(r'(?:^|\s)((?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,})(?:\.\S*)?)', re.IGNORECASE) # Slightly simplified noscheme pattern

    scheme_match = url_pattern_scheme.search(input_text)
    if scheme_match: url_found_info = {"match_str": scheme_match.group(0), "original": scheme_match.group(0)}
    else:
        noscheme_match = url_pattern_noscheme.search(input_text)
        if noscheme_match:
            potential_domain_part = noscheme_match.group(1).strip('.').strip('/') # Get potential domain
            # Basic filter: length check and avoid things obviously not domains
            if '.' in potential_domain_part and len(potential_domain_part) > 3 and not potential_domain_part.lower().endswith(('.png', '.jpg', '.js', '.css', '.py', '.pdf')):
                potential_url = "https://" + potential_domain_part # Default to https
                if is_valid_url(potential_url): url_found_info = {"match_str": potential_url, "original": potential_domain_part}

    if url_found_info:
        try: url = sanitize_url_for_scan(url_found_info["match_str"]) # Sanitize potential URL
        except Exception as e: logger.warning(f"URL sanitization error: {e}"); url = None
        if url: context_text = input_text.replace(url_found_info["original"], "[URL]", 1).strip(); logger.info(f"[ReqID: {request_id}] URL Found & Sanitized: {url}")
        if context_text == "[URL]" or not context_text: context_text = None

    if not url: # If still no valid, sanitized URL
        logger.warning(f"[ReqID: {request_id}] No processable URL found.")
        consolidated = consolidate_url_assessment("N/A", None, {s: ScanResultDetail(status="skipped", details={"message":"No URL found"}) for s in ["vt","ipqs","urlscan"]})
        consolidated.update({"assessment": "Analysis Failed", "analysis_summary": "No URL found or processed.", "evidence_notes": ["No processable URL identified."], "confidence_score": 0.1 })
    else: # URL found, proceed with analysis
        # 2. Text Context
        text_assessment: Optional[TextContextAssessment] = None
        if context_text: text_assessment = _analyze_text_context(request_id, context_text)
        else: text_assessment = TextContextAssessment(suspicion_level="N/A", key_indicators=[])
        # 3. Concurrent Scans
        scan_outputs = await _perform_url_scans(request_id, url)
        # 5. Consolidate
        consolidated = consolidate_url_assessment(url, text_assessment, scan_outputs)

    # Assemble response outside the conditional
    response = UrlAnalysisResponse(request_id=request_id, input_text=input_text, processing_time_ms=0.0, **consolidated)
    logger.info(f"[ReqID: {request_id}] URL Analysis Assessment: {response.assessment}, Confidence: {response.confidence_score}")
    return response

# --- Helper for URL Text Context ---
def _analyze_text_context(request_id: str, context_text: str) -> TextContextAssessment:
    """Analyzes surrounding text for scam/phishing keywords."""
    logger.debug(f"[ReqID: {request_id}] Analyzing text context...")
    suspicion: Literal["High", "Medium", "Low"] = "Low"; indicators = []
    lower_context = context_text.lower()
    high_kws = ["free", "won", "prize", "lottery", "claim", "urgent", "verify", "login", "update", "suspended", "confirm identity", "password", "ssn", "bank", "payment"]
    med_kws = ["delivery", "package", "confirm", "notification", "invoice", "shipment", "alert", "security", "gift card", "refund", "support", "account issue"]
    matched_h = [k for k in high_kws if k in lower_context]; matched_m = [k for k in med_kws if k in lower_context]
    if matched_h: suspicion = "High"; indicators = ["Detected high-risk keywords."] + matched_h
    elif matched_m: suspicion = "Medium"; indicators = ["Detected medium-risk keywords."] + matched_m
    logger.debug(f"Text context assessment: {suspicion}")
    return TextContextAssessment(suspicion_level=suspicion, key_indicators=indicators)

# --- Helper for Concurrent URL Scans ---
async def _perform_url_scans(request_id: str, url: str) -> Dict[str, Optional[ScanResultDetail]]:
    """Runs VT, IPQS, URLScan concurrently and processes results."""
    scan_tasks = { "virustotal": check_virustotal(url), "ipqualityscore": check_ipqs(url), "urlscanio": check_urlscan_existing_results(url) }
    scan_start = time.perf_counter(); results = await asyncio.gather(*scan_tasks.values(), return_exceptions=True)
    logger.debug(f"[ReqID: {request_id}] Scans completed in {time.perf_counter() - scan_start:.3f}s")

    scan_outputs: Dict[str, Optional[ScanResultDetail]] = {}
    parser_map = {"virustotal": parse_vt_result, "ipqualityscore": parse_ipqs_result, "urlscanio": parse_urlscan_result}

    for i, (scanner_name, result_or_exc) in enumerate(zip(scan_tasks.keys(), results)):
        status="error"; details={"message":"Unknown error"}
        if isinstance(result_or_exc, RateLimitException): status="rate_limited"; details={"message":str(result_or_exc)}
        elif isinstance(result_or_exc, ApiException): status="error"; details={"message":str(result_or_exc)}
        elif isinstance(result_or_exc, Exception): status="error"; details={"message":f"Internal: {type(result_or_exc).__name__}"}; logger.error(f"{scanner_name} scan error: {result_or_exc}", exc_info=True)
        elif result_or_exc is None: status="no_data"; details={"message":"No data returned"}
        else: # Success case - try parsing
            parser = parser_map.get(scanner_name)
            if parser:
                try: parsed = parser(result_or_exc); status = parsed.get("status", "error"); details = parsed.get("details", {})
                except Exception as e: status="error"; details={"message":f"Parsing failed: {e}"}; logger.error(f"Parsing {scanner_name} failed: {e}", exc_info=True)
            else: status="error"; details={"message":"No parser defined"}

        scan_outputs[scanner_name] = ScanResultDetail(status=status, details=details)

    # --- Check for critical failures after parsing ---
    successful_or_nodata = [s for s in scan_outputs.values() if s.status in ['success', 'no_data', 'pending', 'no_scan_found']]
    if not successful_or_nodata: # If NO scan yielded any usable info (success or known lack of data)
        rate_limits_hit = any(s.status == 'rate_limited' for s in scan_outputs.values())
        if rate_limits_hit: raise RateLimitException(f"Rate limit reached on scanner(s): {[n for n,s in scan_outputs.items() if s.status=='rate_limited']}")
        else: raise ApiException(f"Critical error or no data from all scanners: {[n for n,s in scan_outputs.items() if s.status=='error']}") # Raise general API Exception if all failed

    return scan_outputs


async def handle_misinfo_analysis(request_id: str, input_text: str) -> MisinformationAnalysisResponse:
    """Handles the Misinformation analysis workflow (RAG/LLM/Web)."""
    logger.info(f"[ReqID: {request_id}] Starting Misinformation analysis workflow.")
    global rag_processor
    if not rag_processor: logger.error(f"[ReqID: {request_id}] RAG processor unavailable."); raise ApiException("RAG processor not available.")

    # Initialize vars
    assessment: Literal["Likely Factual", "Likely Misleading", "Opinion", "Needs Verification / Uncertain", "Contradictory Information Found"] = "Needs Verification / Uncertain"
    confidence = 0.5; explanation = "Analysis pending."; evidence = []; kg_insights = None
    data_source: Literal["RAG", "LLM Internal Knowledge", "Web Search"] = "LLM Internal Knowledge"; extracted_ents = []

    # KG Query
    try: extracted_ents = extract_entities(input_text); kg_insights = query_kg_for_entities(extracted_ents) if extracted_ents else "No relevant entities."
    except Exception as kg_err: logger.error(f"KG query error: {kg_err}", exc_info=True); kg_insights = "KG query failed."

    # --- Orchestration ---
    rag_response, rag_sources = None, None
    try: rag_response, rag_sources = await rag_processor.query_rag(input_text, use_for="misinfo_check") # 1. Try RAG
    except Exception as e: logger.error(f"RAG query exception: {e}", exc_info=True)

    if rag_response and rag_sources: # RAG Path
        data_source = "RAG"; explanation = rag_response
        evidence = [EvidenceItem(source=str(s.get('source','RAG Doc')), snippet=str(s.get('snippet', '')), assessment_note="From RAG") for s in rag_sources]
        logger.info(f"[ReqID: {request_id}] Using RAG result for misinfo.")
        # Simple assessment based on keywords
        lexp = explanation.lower(); assessment, confidence = _assess_llm_response_keywords(lexp)
    else: # Fallback Path
        logger.info(f"[ReqID: {request_id}] RAG insufficient/failed. Trying direct LLM analysis.")
        try: # 2. Try Direct Groq Analysis
            direct_groq_result = await analyze_misinformation_groq(input_text)
            if direct_groq_result and direct_groq_result.get("category") != "error":
                data_source = "LLM Internal Knowledge"; cat = direct_groq_result.get("category", "other")
                map_assessment = {"likely_factual": "Likely Factual", "likely_misleading": "Likely Misleading", "opinion": "Opinion",
                                  "needs_verification": "Needs Verification / Uncertain", "contradictory": "Contradictory Information Found",
                                  "satire": "Opinion", "other": "Needs Verification / Uncertain"}
                assessment = map_assessment.get(cat, "Needs Verification / Uncertain")
                confidence = direct_groq_result.get("confidence", 0.4)
                explanation = direct_groq_result.get("explanation", "") + f" | Issues: {direct_groq_result.get('key_issues', [])}"
                evidence = [EvidenceItem(source="LLM Analysis", snippet=explanation[:200]+"...", assessment_note=f"LLM cat: {cat}")]
                logger.info(f"[ReqID: {request_id}] Direct LLM analysis used. Category: {cat}")
            else: raise GroqApiException("Direct analysis failed/error")
        except Exception as llm_err: # Handle Direct LLM Failure -> Web Search
            logger.warning(f"[ReqID: {request_id}] Direct LLM failed ({llm_err}). Trying web search fallback.")
            try: # 3. Web Search Fallback
                 search_results = await perform_search(input_text)
                 if search_results:
                     data_source = "Web Search"; snippets_text = "\n---\n".join([f"Source: {res['link']}\nSnippet: {res['snippet']}" for res in search_results])
                     synthesis_prompt = f"""Evaluate statement based ONLY on snippets. Focus on support/contradiction. Statement: "{input_text}"\nSnippets:\n{snippets_text}\nEvaluation:"""
                     web_synthesis = await query_groq(synthesis_prompt, temperature=0.1)
                     if web_synthesis:
                         explanation = f"(Web Search) {web_synthesis}"; evidence = [EvidenceItem(source=res['link'], snippet=res['snippet'], assessment_note="From Web Search") for res in search_results]
                         assessment, confidence = _assess_llm_response_keywords(web_synthesis.lower(), base_confidence=0.80) # Use helper, higher base confidence
                         logger.info(f"[ReqID: {request_id}] Web search synthesis successful.")
                     else: explanation="RAG/LLM failed. Web synthesis LLM call failed."; assessment="Needs Verification / Uncertain"; confidence=0.2
                 else: explanation="RAG/LLM failed. Web search found no results."; assessment="Needs Verification / Uncertain"; confidence=0.1
            except Exception as search_err:
                 logger.error(f"Web search fallback failed: {search_err}", exc_info=True); explanation = "Web search failed."; assessment="Needs Verification / Uncertain"; confidence=0.15

    # Update KG
    try:
        if extracted_ents: add_claim_to_graph(input_text, str(assessment), f"user_input:{request_id}", extracted_ents); logger.info("Updated KG.")
    except Exception as e: logger.error(f"KG update failed: {e}", exc_info=True)

    return MisinformationAnalysisResponse(
        request_id=request_id, input_text=input_text, processing_time_ms=0.0, # Set outside
        assessment=assessment, confidence_score=round(confidence, 3), explanation=explanation,
        evidence=evidence, knowledge_graph_insights=kg_insights )


async def handle_factual_analysis(request_id: str, input_text: str) -> FactualAnalysisResponse:
    """Handles the Factual QA workflow (RAG/LLM/Web)."""
    logger.info(f"[ReqID: {request_id}] Starting Factual analysis workflow.")
    global rag_processor
    if not rag_processor: logger.error(f"[ReqID: {request_id}] RAG unavailable."); raise ApiException("RAG processor not available.")

    # Initialize vars
    assessment: Literal["Likely Factual", "Opinion", "Needs Verification / Uncertain", "Contradictory Information Found"] = "Needs Verification / Uncertain"
    confidence=0.5; answer="Analysis pending."; explanation=None; evidence=[]; kg_insights=None
    data_source: Literal["RAG", "LLM Internal Knowledge", "Web Search"] = "LLM Internal Knowledge"; extracted_ents = []

    # KG Query (Optional)
    try: extracted_ents = extract_entities(input_text); kg_insights = "KG available (not queried for factual)."
    except Exception as e: logger.error(f"KG extraction error: {e}", exc_info=True)

    # --- Orchestration ---
    rag_response, rag_sources = None, None
    # 1. Try RAG
    try: rag_response, rag_sources = await rag_processor.query_rag(input_text, use_for="factual_qa")
    except Exception as e: logger.error(f"RAG query exception: {e}", exc_info=True)

    if rag_response and rag_sources: # RAG Path
        data_source="RAG"; answer=rag_response
        evidence=[EvidenceItem(source=str(s.get('source','RAG Doc')), snippet=str(s.get('snippet','')), assessment_note="From RAG") for s in rag_sources]
        logger.info(f"[ReqID: {request_id}] Using RAG for factual.")
        l_ans = answer.lower()
        if any(w in l_ans for w in ["don't know", "context doesn't", "cannot answer"]): assessment="Needs Verification / Uncertain"; confidence=0.4
        elif "opinion" in l_ans and len(answer)<250: assessment="Opinion"; confidence=0.7
        else: assessment="Likely Factual"; confidence=0.88
    else: # Fallback Path
        logger.info(f"[ReqID: {request_id}] RAG failed/insufficient. Trying direct LLM.")
        try: # 2. Try Direct LLM
            direct_groq_result = await ask_groq_factual(input_text)
            if direct_groq_result and direct_groq_result.get("confidence_level") != "error":
                data_source = "LLM Internal Knowledge"; answer = direct_groq_result.get("answer", ""); g_conf = direct_groq_result.get("confidence_level", "low")
                evidence=[EvidenceItem(source="LLM Analysis", snippet=answer[:200]+"...", assessment_note=f"LLM conf: {g_conf}")]
                map={"high": 0.85, "medium": 0.65, "low": 0.35}; confidence=map.get(g_conf, 0.3)
                assessment = "Needs Verification / Uncertain" if g_conf == "low" else "Likely Factual"
                logger.info(f"[ReqID: {request_id}] Direct LLM factual Q&A successful. LLM Conf: {g_conf}")
            else: raise GroqApiException("Direct factual returned error/no answer")
        except Exception as llm_err: # Handle LLM Fail -> Web Search
            logger.warning(f"[ReqID: {request_id}] Direct LLM failed ({llm_err}). Trying web fallback.")
            try: # 3. Web Search Fallback
                 search_results = await perform_search(input_text)
                 if search_results:
                     data_source="Web Search"; snippets="\n---\n".join([f"Source:{r['link']}\n{r['snippet']}" for r in search_results])
                     s_prompt = f"Answer question based ONLY on snippets:\nQ: \"{input_text}\"\nSnippets:\n{snippets}\nAnswer (state if cannot answer):"
                     web_synth = await query_groq(s_prompt, temperature=0.05)
                     if web_synth and not any(w in web_synth.lower() for w in ["cannot answer", "not found", "unable to determine"]):
                          answer=web_synth; evidence=[EvidenceItem(source=r['link'], snippet=r['snippet'], assessment_note="From Web Search") for r in search_results]
                          assessment="Likely Factual"; confidence=0.85; logger.info(f"[ReqID: {request_id}] Web search synthesis provided answer.")
                     else: logger.warning(f"Web search synthesis failed or inconclusive."); answer="Could not verify answer via web search."; assessment="Needs Verification / Uncertain"; confidence=0.3
                 else: logger.warning(f"Web search fallback found no results."); answer="RAG/LLM/Web Search failed."; assessment="Needs Verification / Uncertain"; confidence=0.2
            except Exception as search_err:
                 logger.error(f"Web search fallback failed: {search_err}", exc_info=True); answer="Web search failed."; assessment="Needs Verification / Uncertain"; confidence=0.15

    return FactualAnalysisResponse( request_id=request_id, input_text=input_text, processing_time_ms=0.0, # Set outside
        assessment=assessment, answer=answer, explanation=explanation, confidence_score=round(confidence, 3),
        data_source=data_source, supporting_evidence=evidence, knowledge_graph_insights=kg_insights )

# --- Helper to assess LLM response ---
# In api/main.py
def _assess_llm_response_keywords(text: str, base_confidence: float = 0.6) -> Tuple[Literal["Likely Factual", "Likely Misleading", "Opinion", "Needs Verification / Uncertain", "Contradictory Information Found"], float]:
    """Placeholder assessment based on keywords in LLM explanation/synthesis."""
    lower_text = text.lower()
    confidence_bonus = 0.0
    assessment: Literal["Likely Factual", "Likely Misleading", "Opinion", "Needs Verification / Uncertain", "Contradictory Information Found"] = "Needs Verification / Uncertain"

    if any(w in lower_text for w in ["contradicts", "inaccurate", "false", "misleading", "refuted", "not supported"]):
        assessment = "Likely Misleading"; confidence_bonus = 0.25
    elif any(w in lower_text for w in ["supported", "accurate", "confirmed", "factual", "aligns with"]):
        assessment = "Likely Factual"; confidence_bonus = 0.25
    elif any(w in lower_text for w in ["opinion", "subjective", "viewpoint"]):
        assessment = "Opinion"; confidence_bonus = 0.15
    elif any(w in lower_text for w in ["conflicting", "contradictory", "disagree", "mixed information"]):
        assessment = "Contradictory Information Found"; confidence_bonus = 0.10
    elif any(w in lower_text for w in ["insufficient", "cannot determine", "not addressed", "unclear", "speculative"]):
        assessment = "Needs Verification / Uncertain"; confidence_bonus = -0.10 # Negative bonus for uncertainty
    else:
        assessment = "Needs Verification / Uncertain"; confidence_bonus = 0.0 # Default if no strong keywords

    # Calculate final confidence and CLAMP it between 0.05 and 0.99
    final_confidence = min(max(base_confidence + confidence_bonus, 0.05), 0.99) # Ensure score stays in valid range

    return assessment, final_confidence # Return the clamped confidence
