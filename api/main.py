# api/main.py
import os
import json
import re
import logging
import hashlib
import yaml
import asyncio
import redis.asyncio as aioredis  # Use redis' asyncio module
# Ensure this is imported if not already

from fastapi import FastAPI, HTTPException, Request, Depends, Security # Keep Depends, Security even if unused now
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel # Use BaseModel directly here
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List

# --- Relative Imports for API modules ---
from .utils import is_valid_url, ping_url, sanitize_url_for_virustotal
from .groq_utils import analyze_misinformation_groq, ask_groq_factual
from .vt_utils import check_url_safety
from .langchain_utils import RealTimeDataProcessor
from .classifier import load_classifier_model, classify_query_local # Local classifier
# --- Import from project root ---
# Make sure this path works based on how you run uvicorn
# If running from the root 'misinformation_detection' dir:
from rss_feed_ingestor import get_rss_news
# If running uvicorn inside the 'api' dir, it might need adjustment,
# but running 'uvicorn api.main:app' from the root is standard.

# --- Redis Cache ---
from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache
from fastapi_cache.backends.redis import RedisBackend
# from redis.asyncio import Redis # No need to import Redis directly if using aioredis.from_url

# --- Ensure logs directory exists ---
if not os.path.exists("logs"):
    os.makedirs("logs")
    print("Created 'logs' directory.") # Simple print for confirmation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)-4d | %(message)s', # Enhanced format
    handlers=[
        logging.FileHandler("logs/api.log"), # Log file in the logs directory
        logging.StreamHandler()
    ],
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- API Key Setup (Keep definition, but disable usage on endpoint for now) ---
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
EXPECTED_API_KEY = os.getenv("API_KEY")

async def get_api_key(key: str = Security(api_key_header)):
    """Dependency to validate the API key (Not actively used on /analyze/ currently)."""
    if not EXPECTED_API_KEY:
         logger.warning("API_KEY environment variable is not set. Denying access.")
         raise HTTPException(status_code=403, detail="Access denied: Server configuration issue.")

    if key == EXPECTED_API_KEY:
        return key
    else:
        logger.warning(f"Invalid API Key received: {key[:4]}...")
        raise HTTPException(
            status_code=403,
            detail="Could not validate credentials",
        )

# --- Define Pydantic Models Directly Here ---
class QueryModel(BaseModel):
    question: str

class AnalysisResponse(BaseModel):
    result: Dict[str, Any]
    analysis_type: str
    status: str
    message: Optional[str] = None


# Initialize FastAPI app
app = FastAPI(
    title="Misinformation Detection API",
    description="Enhanced pipeline using RAG, Local Classification, VirusTotal, Groq, Redis Cache. (API Key Temp Disabled)", # Updated description
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", API_KEY_NAME], # Still allow header, even if check is disabled
)

# Load configuration
def load_app_config(config_path="config/config.yaml"):
    """Load configuration from the YAML file with error handling."""
    try:
        with open(config_path, 'r') as f:
            conf = yaml.safe_load(f)
            if not isinstance(conf, dict):
                 raise yaml.YAMLError("Config file is not a valid dictionary.")
            # --- YAML Syntax Correction Check (Keep if needed) ---
            if 'data_sources' in conf and isinstance(conf['data_sources'], dict):
                corrected_sources = {}
                for k, v in conf['data_sources'].items():
                     corrected_key = k.strip().strip('"')
                     corrected_sources[corrected_key] = v
                conf['data_sources'] = corrected_sources
            # --- End Correction ---
            logger.info(f"Configuration loaded successfully from {config_path}")
            return conf
    except FileNotFoundError:
        logger.error(f"FATAL: Configuration file not found at {config_path}. Exiting.")
        raise SystemExit(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        logger.error(f"FATAL: Error parsing configuration file {config_path}: {e}. Exiting.")
        raise SystemExit(f"Invalid configuration file: {e}")

config = load_app_config()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/rag_data/default_index")

# --- Global Instances (Initialized during startup) ---
rag_processor: Optional[RealTimeDataProcessor] = None # Added Optional typing hint

# --- Startup and Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    """Tasks to run when the application starts."""
    global rag_processor

    # Ensure logs dir exists (redundant check, but safe)
    if not os.path.exists("logs"):
        try:
            os.makedirs("logs")
            logger.info("Created 'logs' directory during startup.")
        except OSError as e:
            logger.error(f"Could not create 'logs' directory: {e}")


    # 1. Load Classifier Model
    logger.info("Loading local classification model...")
    load_classifier_model()

    # 2. Initialize RAG Processor
    logger.info("Initializing RAG processor...")
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not found. RAG processor query functionality will fail.")
    # RAG processor needs Groq key to query LLM after retrieval
    rag_processor = RealTimeDataProcessor(api_key=GROQ_API_KEY, index_path=FAISS_INDEX_PATH)

    # 3. Initialize Redis Cache
    logger.info("Initializing Redis Cache...")
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = os.getenv("REDIS_PORT", "6379")
    redis_db = os.getenv("REDIS_DB", "0")
    redis_password = os.getenv("REDIS_PASSWORD", None) or None
    redis_url = f"redis://{':' + redis_password + '@' if redis_password else ''}{redis_host}:{redis_port}/{redis_db}"

    try:
        # Use aioredis directly
        redis_client = aioredis.from_url(redis_url, encoding="utf8", decode_responses=True)
        await redis_client.ping()
        FastAPICache.init(RedisBackend(redis_client), prefix="fastapi-cache")
        logger.info(f"Redis cache connected successfully to {redis_host}:{redis_port}/{redis_db}")
    except Exception as e:
         logger.error(f"Failed to connect to Redis at {redis_url}: {e}. Caching will be disabled.", exc_info=True)
         # Allow app to start without cache

    # 4. Initial Data Ingestion (Run in background task)
    logger.info("Scheduling initial data ingestion...")
    asyncio.create_task(scheduled_data_ingestion()) # No longer need background thread wrapper
    logger.info("Startup sequence complete. API is ready.")


async def scheduled_data_ingestion():
    """Fetches RSS news and updates the RAG index asynchronously."""
    await asyncio.sleep(5)
    logger.info("Starting scheduled data ingestion task...")

    rss_feeds = config.get('data_sources', {})
    if not rss_feeds:
        logger.warning("No data sources configured in config.yaml for ingestion.")
        return

    # Fetch news concurrently
    tasks = [asyncio.create_task(fetch_and_process_feed(source, url)) for source, url in rss_feeds.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_articles = []
    for result in results:
        if isinstance(result, list):
            all_articles.extend(result)
        elif isinstance(result, Exception):
            logger.error(f"Error during concurrent feed fetching: {result}", exc_info=result)

    if all_articles:
        logger.info(f"Total articles fetched: {len(all_articles)}. Updating RAG index...")
        if rag_processor:
             try:
                 # Use asyncio.to_thread as update_index itself might be blocking
                 await asyncio.to_thread(rag_processor.update_index, all_articles)
                 logger.info("RAG index update task submitted.")
             except Exception as e:
                 logger.error(f"Error submitting RAG index update: {e}", exc_info=True)
        else:
            logger.error("Cannot update RAG index: RAG processor not initialized.")
    else:
        logger.warning("No articles collected to update RAG index.")

    logger.info("Scheduled data ingestion task finished.")


async def fetch_and_process_feed(source: str, feed_url: str) -> Optional[List[str]]:
    """Helper for concurrent fetching using asyncio.to_thread."""
    logger.info(f"Fetching feed: {source}")
    try:
        processed_feed_url = feed_url # Handle potential formatting later if needed
        if "{query}" in feed_url:
            default_query = "world+news"
            # Example query override logic (can be expanded)
            rss_query_override = config.get('rss_query_override', {})
            query = rss_query_override.get(source, default_query)
            processed_feed_url = feed_url.format(query=query.replace(" ", "+"))
            logger.info(f"Using query '{query}' for {source} URL: {processed_feed_url}")
        else:
             logger.info(f"Using static URL for {source}: {processed_feed_url}")


        # Run the blocking feedparser in a thread
        articles = await asyncio.to_thread(
             get_rss_news, # The function to run
             processed_feed_url, # Argument 1 for get_rss_news
             config.get('rss_max_articles_per_source', 10) # Argument 2, get limit from config
        )

        if articles:
            logger.info(f"Successfully ingested {len(articles)} articles from {source}")
            return articles
        else:
            logger.warning(f"No articles ingested from {source}")
            return None # Explicitly return None if no articles
    except Exception as e:
        logger.error(f"Failed to ingest from {source} ({feed_url}): {e}", exc_info=True)
        return None # Return None on error


# --- API Endpoints ---
@app.get("/", tags=["General"])
def home():
    """Basic API information and status."""
    cache_backend = FastAPICache.get_backend()
    redis_status = "Not Initialized"
    if cache_backend and hasattr(cache_backend, 'redis') and cache_backend.redis:
        # Basic check: assumes connection is okay if client exists post-startup
        redis_status = "Connected" # Or add a ping check here if needed often
    elif cache_backend:
        redis_status = "Initialized (Unknown State)"


    api_status = {
        "groq_api": "Available" if GROQ_API_KEY else "MISSING",
        "virustotal_api": "Available" if VIRUSTOTAL_API_KEY else "MISSING",
        "local_classifier": "Loaded" if callable(classify_query_local) and globals().get('classifier') else "Not Loaded",
        "rag_db": "Loaded" if rag_processor and rag_processor.db else "Not Loaded/Available",
        "redis_cache": redis_status
    }
    return {
        "message": "Misinformation Detection API is running!",
        "version": app.version,
        "api_status": api_status,
        "endpoints": {
            "/analyze/": "POST - Analyze input (API Key Check Temporarily Disabled)", # Updated message
            "/health/": "GET - Health check endpoint"
        }
    }

@app.get("/health/", tags=["General"])
async def health_check(): # Made async to potentially check async dependencies like Redis ping
    """Simple health check, potentially checks critical async dependencies."""
    # Optional: Add an async check here, e.g., redis ping
    try:
        cache_backend = FastAPICache.get_backend()
        if cache_backend and hasattr(cache_backend, 'redis') and cache_backend.redis:
            await cache_backend.redis.ping()
        # Add other checks if needed (e.g., can RAG processor retrieve?)
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


# --- Main Analysis Endpoint ---
@app.post("/analyze/",
          response_model=AnalysisResponse,
          tags=["Analysis"]
          # dependencies=[Depends(get_api_key)] # <<< TEMPORARILY DISABLED FOR TESTING >>>
         )
@cache(expire=300) # Cache results for 5 minutes (300 seconds)
async def analyze_input(query: QueryModel):
    """
    Analyzes input text for misinformation, factual QA, or URL safety.
    (API Key check temporarily disabled for development/testing)
    """
    if not query.question or not query.question.strip():
         raise HTTPException(status_code=400, detail="Input question cannot be empty.")

    input_text = query.question.strip()
    logger.info(f"Received query: '{input_text[:100]}...'")

    if not rag_processor:
        logger.error("RAG processor is not available. Cannot perform RAG-dependent analyses.")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable (Knowledge Base).")
    if not GROQ_API_KEY:
         logger.error("Groq API Key missing. Cannot perform Groq-dependent analyses.")
         raise HTTPException(status_code=503, detail="Service temporarily unavailable (Configuration Error).")
    if not VIRUSTOTAL_API_KEY:
         logger.warning("VirusTotal API Key missing. URL safety checks will fail.")
         # Allow to proceed but log warning - URL checks will return error


    # 1. Classify the query type using LOCAL model
    # Add try-except block around classification in case model fails badly
    try:
        analysis_type = classify_query_local(input_text)
    except Exception as e:
        logger.error(f"Local classification failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error: Failed to classify input.")

    logger.info(f"Classified input type as: {analysis_type}")

    result: Dict[str, Any] = {}
    status = "error"
    message = None
    groq_model = config.get('groq', {}).get('model', 'mixtral-8x7b-32768') # Default if not in config

    try:
        if analysis_type == "url":
            if not is_valid_url(input_text):
                 message = "Input does not appear to be a processable URL."
                 result = {"error": message}
                 status = "invalid_input"
            else:
                 sanitized = sanitize_url_for_virustotal(input_text)
                 is_reachable = await ping_url(sanitized)
                 if not is_reachable:
                     message = "URL appears to be unreachable or does not exist."
                     result = {"error": message, "url": sanitized}
                     status = "unreachable_url"
                 else:
                     if not VIRUSTOTAL_API_KEY:
                          result = {"error": "VirusTotal API Key not configured.", "url": sanitized}
                          status = "error"
                          message = "URL safety check unavailable due to missing configuration."
                     else:
                          logger.info(f"Checking safety with VirusTotal: {sanitized}")
                          result = await check_url_safety(sanitized, VIRUSTOTAL_API_KEY)
                          status = "success" if "error" not in result else "error"
                          if result.get("status") == "pending" or result.get("status") == "submitted":
                              status = result["status"]
                              message = result.get("message", "VirusTotal analysis processing.")

        elif analysis_type == "misinfo":
            logger.info("Performing RAG + Groq misinformation check...")
            rag_result = await rag_processor.query_rag(input_text, groq_model, use_for="misinfo_check")

            if "error" not in rag_result and rag_result.get("source") != "RAG - No Context Found" and rag_result.get("source") != "RAG Error - No DB":
                 logger.info("RAG provided analysis based on context.")
                 result = rag_result
                 status = "success"
            else:
                 logger.warning(f"RAG insufficient/failed ({rag_result.get('source', 'Unknown RAG issue')}), falling back to Groq internal knowledge.")
                 result = await analyze_misinformation_groq(input_text, GROQ_API_KEY, groq_model)
                 status = "success" if "error" not in result else "error"
                 if status == "success":
                      result["note"] = "Analysis based on general knowledge, not real-time context."
                      result["source"] = "Groq Internal Knowledge" # Clarify fallback source

        elif analysis_type == "factual":
            logger.info("Performing Groq factual QA with potential RAG fallback...")
            groq_answer = await ask_groq_factual(input_text, GROQ_API_KEY, groq_model)

            needs_rag_fallback = False
            if "error" in groq_answer:
                needs_rag_fallback = True
                logger.warning("Groq factual QA failed, attempting RAG fallback.")
            elif groq_answer.get("confidence_level") == "medium":
                 if any(phrase in groq_answer.get("answer", "").lower() for phrase in ["knowledge cutoff", "recent events", "limited data after", "may be outdated", "cannot provide real-time"]):
                    needs_rag_fallback = True
                    logger.info("Groq answer suggests limited recent knowledge, attempting RAG fallback.")

            if needs_rag_fallback:
                logger.info("Attempting RAG fallback for factual query...")
                rag_fallback_result = await rag_processor.query_rag(input_text, groq_model, use_for="factual_fallback")
                if rag_fallback_result and "error" not in rag_fallback_result and rag_fallback_result.get("source") != "RAG - No Context Found":
                     logger.info("RAG fallback provided an answer.")
                     result = rag_fallback_result
                     status = "success"
                     result["note"] = "Answer augmented or replaced using recent context." # Clarify source
                else:
                     logger.warning(f"RAG fallback failed or found no context ({rag_fallback_result.get('source', 'Unknown RAG issue')}). Returning original Groq result.")
                     result = groq_answer
                     status = "success" if "error" not in result else "error"
            else:
                logger.info("Using Groq's primary answer for factual query.")
                result = groq_answer
                status = "success" if "error" not in result else "error"

        else: # Includes 'other' or unexpected classification result
            logger.warning(f"Input classified as '{analysis_type}'. Handling as general Groq query.")
            # Fallback to a general Groq query if classification is uncertain or 'other'
            result = await ask_groq_factual(input_text, GROQ_API_KEY, groq_model) # Use factual QA as a default
            status = "success" if "error" not in result else "error"
            analysis_type = "general_query" # Re-label type for clarity in response


        # Final check for errors within the result dict
        if isinstance(result, dict) and "error" in result and not message:
             message = result["error"]
             # Ensure status reflects error if not already set
             if status != "error" and status != "invalid_input" and status != "unreachable_url" and status != "pending" and status != "submitted":
                 status = "error"


        # Add Groq model used to the result for transparency
        if isinstance(result, dict):
            result["llm_model_used"] = groq_model

        return AnalysisResponse(
            result=result,
            analysis_type=analysis_type,
            status=status,
            message=message
        )

    except HTTPException as e:
         raise e # Re-raise FastAPI's managed exceptions
    except Exception as e:
         logger.error(f"Unexpected error during analysis of '{input_text[:50]}...': {str(e)}", exc_info=True)
         # Construct a response manually for unexpected errors
         return AnalysisResponse(
             result={"error": f"An unexpected server error occurred."},
             analysis_type=analysis_type if 'analysis_type' in locals() else "unknown",
             status="error",
             message=f"Unexpected error processing request: {str(e)}" # Provide error detail in message
         )

# --- Run Application ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server directly (for debugging)...")
    # Standard command: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True) # Pass app as string for reload
