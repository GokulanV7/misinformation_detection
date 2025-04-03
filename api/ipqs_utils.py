# api/ipqs_utils.py
import logging
from typing import Optional, Dict, Any
import httpx
from dotenv import load_dotenv
import os
import urllib.parse # <-- Ensure imported
import json        # <-- Ensure imported for JSONDecodeError

# --- Custom Exception Imports ---
try:
    from .utils import get_config, RateLimitException, ApiException
except ImportError:
    print("Warning: Running ipqs_utils possibly standalone.")
    from utils import get_config, RateLimitException, ApiException # type: ignore

# --- Setup ---
load_dotenv()
CONFIG = get_config()
IPQS_CONFIG = CONFIG.get('ipqualityscore', {}) # Use .get() for safety
API_KEY = os.getenv("IPQUALITYSCORE_API_KEY")
# Use url_check_api_url from config, ensure it's correct base URL ending with /
API_BASE_URL = IPQS_CONFIG.get('url_check_api_url', "https://ipqualityscore.com/api/json/url/") # Default fallback
TIMEOUT = IPQS_CONFIG.get('request_timeout', 20)
logger = logging.getLogger(__name__)

async def check_ipqs(url: str) -> Optional[Dict[str, Any]]:
    """
    Checks a URL's reputation using the IPQualityScore API.

    Returns: Raw JSON response dict on success, None on 404, timeout, network error,
             or parsing failure.
    Raises: RateLimitException, ApiException for relevant API errors.
    """
    if not API_KEY:
        logger.error("IPQualityScore API key not found in env variables.")
        raise ApiException("IPQualityScore API Key is not configured.") # Raise for config issue
    if not url:
         logger.warning("IPQS check called with empty URL.")
         return None

    try:
        # URL Encode the target URL as it's part of the path in their API structure
        encoded_target_url = urllib.parse.quote(url.strip(), safe='')
        # Construct the FULL request URL including the key and target URL
        ipqs_request_url = f"{API_BASE_URL}{API_KEY}/{encoded_target_url}"

        # Parameters likely go in query string - check IPQS docs for specifics
        # Common ones: strictness, allow_public_access_points, fast, timeout
        ipqs_params = {
            'strictness': '1',  # Example: Higher strictness
            'allow_public_access_points': 'true', # Example
            'fast': 'false',    # Example: Get more details, might be slower
            'timeout': str(TIMEOUT * 900) # Send timeout in milliseconds
        }
        ipqs_params = {k: v for k, v in ipqs_params.items() if v is not None} # Clean params

        logger.debug(f"Sending request to IPQS: URL={ipqs_request_url}, Params={ipqs_params}")

        async with httpx.AsyncClient(timeout=TIMEOUT + 5) as client:
            response = await client.get(ipqs_request_url, params=ipqs_params)

            # --- Specific Error Checks ---
            if response.status_code == 429:
                logger.warning(f"IPQS rate limit (429) for {url}.")
                raise RateLimitException("IPQS rate limit reached.")
            if response.status_code == 401:
                 logger.error(f"IPQS unauthorized (401) for {url}. Check API Key.")
                 raise ApiException("IPQS API Key invalid/misconfigured.")
            if response.status_code == 404: # Handle 404 specifically
                 logger.info(f"IPQS returned 404 Not Found for {url}. Treating as no data.")
                 return None # Indicate no data found
            if response.status_code >= 400: # Other API errors
                 error_msg = f"IPQS API error {response.status_code}"; details = response.text
                 try: details = response.json(); error_msg += f": {details}"
                 except json.JSONDecodeError: error_msg += f": {details}" # Keep text if not JSON
                 logger.error(f"{error_msg} for URL: {url}")
                 raise ApiException(error_msg)
            # --- End Checks ---

            # --- Attempt JSON parsing only on success ---
            try:
                 response_json = response.json()
            except json.JSONDecodeError as json_err:
                 logger.error(f"IPQS returned non-JSON success response ({response.status_code}). Body: '{response.text[:200]}...'. Error: {json_err}")
                 # Treat unexpected format as failure
                 return None

            # Optional: Check for success flag within the JSON response if API uses it
            if not response_json.get('success', True): # Assume success if flag missing, check if False explicitly
                 msg = response_json.get('message', 'Unknown reason.')
                 logger.warning(f"IPQS API call reported unsuccessful: {msg} for {url}")
                 # Decide if this constitutes an API error or just 'no data'
                 if "limit" in msg.lower(): raise RateLimitException(f"IPQS: {msg}")
                 # Otherwise might return None or raise specific exception
                 return None

            logger.debug(f"IPQS successful response for {url}: {str(response_json)[:200]}...")
            return response_json # Return the parsed JSON dict

    # --- Handle Transport / Unexpected Errors ---
    except httpx.TimeoutException:
        logger.warning(f"IPQS request timed out for {url}.")
        return None
    except httpx.RequestError as req_err:
        logger.error(f"Network error contacting IPQS for {url}: {req_err}")
        return None
    except (RateLimitException, ApiException) as e: # Re-raise handled exceptions
        raise e
    except Exception as e: # Catch any other unexpected error
        logger.error(f"Unexpected error during IPQS check for {url}: {e}", exc_info=True)
        return None # Return None for safety

def parse_ipqs_result(ipqs_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Parses the raw IPQS JSON data into a structured format for the response."""
    if not ipqs_data or not isinstance(ipqs_data, dict):
        return {"status": "error", "details": {"message": "Invalid or missing IPQS data object."}}

    # Assume 'success' flag check happened before calling parser if applicable
    # But still check essential fields

    risk = ipqs_data.get('risk_score', -1) # Use -1 to indicate missing score
    category = ipqs_data.get('category', 'N/A')
    message = ipqs_data.get('message', '') # Get message for context

    # Use thresholds from config safely
    high_risk_thr = IPQS_CONFIG.get('high_risk_threshold', 85)
    medium_risk_thr = IPQS_CONFIG.get('medium_risk_threshold', 60)

    assessment = "low_risk" # Default if score exists and is below thresholds
    if risk >= high_risk_thr: assessment = "high_risk"
    elif risk >= medium_risk_thr: assessment = "medium_risk"
    elif risk == -1: assessment = "unknown" # If score was missing

    # Standardize age format if possible
    domain_age = ipqs_data.get('domain_age', {})
    age_days_str = "N/A"
    if domain_age and isinstance(domain_age, dict):
        if 'human' in domain_age: age_days_str = domain_age['human']
        elif 'timestamp' in domain_age: # Calculate from timestamp if needed
            try:
                import datetime
                age_ts = int(domain_age['timestamp'])
                days = (datetime.datetime.now().timestamp() - age_ts) / (60*60*24)
                age_days_str = f"{int(days)} days"
            except: pass # Ignore errors calculating from timestamp


    details_dict = {
        "risk_score": risk if risk != -1 else None, # Return None if missing
        "assessment_category": assessment,
        "threat_category": category,
        "is_phishing": ipqs_data.get('phishing'), # Keep boolean/None
        "is_malware": ipqs_data.get('malware'),
        "is_spam": ipqs_data.get('spamming'),
        "is_suspicious": ipqs_data.get('suspicious'),
        "is_adult": ipqs_data.get('adult'),
        "domain_age_description": age_days_str,
        # Add other useful fields directly
        "dns_valid": ipqs_data.get("dns_valid"),
        "parking": ipqs_data.get("parking"),
        "status_code": ipqs_data.get("status_code"), # From IPQS crawl
        "server": ipqs_data.get("server"),
        # Include raw message if informative
        "api_message": message if message and message != "Success" else None
    }

    # Filter out None values from details for cleaner output
    filtered_details = {k: v for k, v in details_dict.items() if v is not None and v != 'N/A'}

    return {
        "status": "success", # Parsing was successful based on input data
        "details": filtered_details
    }
