# api/vt_utils.py
import logging
import httpx
import os
import time
import asyncio
from base64 import urlsafe_b64encode
from typing import Dict, Any, Optional

from dotenv import load_dotenv

# --- Custom Exception Imports ---
try:
    from .utils import get_config, RateLimitException, ApiException, sanitize_url_for_scan
except ImportError:
    print("Warning: Running vt_utils possibly standalone. Trying relative path for utils.")
    from utils import get_config, RateLimitException, ApiException, sanitize_url_for_scan # type: ignore

# --- Setup ---
load_dotenv()
logger = logging.getLogger(__name__) # Standard logger name
CONFIG = get_config()
VT_CONFIG = CONFIG.get('virustotal', {}) # Use .get() for safety
API_KEY = os.getenv("VIRUSTOTAL_API_KEY")
# API V3 uses different base URLs for different endpoints typically
# Base for URLs endpoint:
URLS_ENDPOINT_BASE = VT_CONFIG.get('api_url', "https://www.virustotal.com/api/v3/urls")
# Base for analyses endpoint (if polling were used):
# ANALYSES_ENDPOINT_BASE = "https://www.virustotal.com/api/v3/analyses"
REQUEST_TIMEOUT = VT_CONFIG.get('request_timeout', 20)
MALICIOUS_THRESHOLD = VT_CONFIG.get('malicious_threshold', 3) # Use from config


# --- Helper Function for VT Requests ---
async def _make_vt_request(method: str, endpoint_url: str, headers: Dict[str, str],
                          params: Optional[Dict] = None, data: Optional[Dict] = None,
                          json_payload: Optional[Dict] = None) -> httpx.Response:
    """
    Makes an async request to a VirusTotal endpoint and handles common errors.
    Raises RateLimitException or ApiException on relevant errors.
    """
    # Note: Using temporary clients here for simplicity, but a shared client is also possible.
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT + 5, headers=headers) as client:
        request_start_time = time.monotonic()
        logger.debug(f"Sending {method} request to VirusTotal: {endpoint_url}")
        response = None
        try:
            if method.upper() == 'GET':
                response = await client.get(endpoint_url, params=params)
            elif method.upper() == 'POST':
                # VT POST /urls expects x-www-form-urlencoded *data*, not JSON
                response = await client.post(endpoint_url, data=data, json=json_payload)
            else:
                raise ValueError(f"Unsupported HTTP method for VT: {method}")

            request_duration = time.monotonic() - request_start_time
            logger.debug(f"VirusTotal response received in {request_duration:.3f}s. Status: {response.status_code}")

            # --- Specific Error Checks ---
            if response.status_code == 429:
                logger.warning(f"VirusTotal rate limit hit (Status 429) for {endpoint_url}")
                raise RateLimitException("VirusTotal API rate limit exceeded.")
            if response.status_code == 401: # Unauthorized
                logger.error("VirusTotal API Unauthorized (Status 401). Check API Key.")
                raise ApiException("VirusTotal API key is invalid or expired.")
            # Check for other 4xx/5xx errors specifically
            if response.status_code >= 400:
                error_message = f"VirusTotal API error {response.status_code}"
                try: error_details = response.json() # VT usually returns JSON errors
                except Exception: error_details = response.text
                error_message += f": {error_details}"
                logger.error(error_message)
                raise ApiException(error_message)
            # --- End Error Checks ---

            return response # Return successful response object

        # Handle client-side/network errors separately
        except httpx.TimeoutException:
            logger.warning(f"VirusTotal request timed out for {endpoint_url}")
            raise ApiException(f"VirusTotal request timed out ({REQUEST_TIMEOUT}s).")
        except httpx.RequestError as req_err:
            logger.error(f"Network error contacting VirusTotal for {endpoint_url}: {req_err}")
            raise ApiException(f"Network error contacting VirusTotal: {req_err}")


# --- Main Function to Check URL ---
async def check_virustotal(url: str) -> Optional[dict]:
    """
    Checks a URL against VirusTotal API v3. Gets report ONLY. Does NOT submit or poll.

    Args:
        url: The URL to check (should be pre-sanitized).

    Returns:
        Dictionary with VT analysis results ('attributes', 'type', 'id') if report exists,
        otherwise None (including cases of 404 Not Found, errors, or no API key).
    Raises:
        RateLimitException: If VT API rate limit is hit.
        ApiException: For other VT API errors or configuration issues.
    """
    if not API_KEY:
        logger.error("VIRUSTOTAL_API_KEY not set. Cannot check VirusTotal.")
        # Application should handle this state gracefully, raising makes sense
        raise ApiException("VirusTotal API Key not configured.")

    # Ensure URL is reasonably clean before generating ID
    # url_sanitized = sanitize_url_for_scan(url) # Assuming sanitization happens *before* this call now

    # URL ID for VT is base64 of the *exact* URL string VT expects
    # Using the passed 'url' directly, assuming it's correctly formatted.
    try:
         url_id = urlsafe_b64encode(url.encode()).decode().strip("=")
    except Exception as e:
         logger.error(f"Failed to generate VirusTotal URL ID for {url}: {e}")
         raise ValueError(f"Invalid URL format for VirusTotal ID generation: {url}")


    headers = {"x-apikey": API_KEY, "Accept": "application/json"}
    report_endpoint_url = f"{URLS_ENDPOINT_BASE}/{url_id}" # V3 format

    try:
        logger.info(f"Checking VirusTotal report for URL ID: {url_id} (URL: {url[:60]}...)")
        response = await _make_vt_request('GET', report_endpoint_url, headers=headers)

        # Only process successful 200 responses
        if response.status_code == 200:
            report_data = response.json()
            # Check if the report contains actual analysis results
            attributes = report_data.get("data", {}).get("attributes", {})
            stats = attributes.get("last_analysis_stats", {})
            if not stats or sum(stats.values()) == 0:
                 logger.info(f"VirusTotal report found for {url_id}, but analysis appears incomplete or pending.")
                 # Treat pending analysis same as not found for immediate checks
                 return None # Return None if analysis isn't ready
            else:
                 logger.info(f"Found existing completed VirusTotal analysis for {url_id}.")
                 return report_data.get("data") # Return the main 'data' object containing attributes etc.

        # Should not be reached due to error checks in _make_vt_request, but as safety:
        logger.warning(f"Unexpected status code {response.status_code} fetching VT report for {url_id}. Treating as 'not found'.")
        return None


    except ApiException as e:
         # Specific check for 404 Not Found, which is *not* an error in this context
         if "404" in str(e):
              logger.info(f"URL not found in VirusTotal database: {url_id} (URL: {url[:60]}...).")
              return None # 404 means no report exists, return None
         else:
              # Re-raise other API errors (401, 429, 5xx, etc.)
              logger.error(f"VirusTotal API Error fetching report for {url}: {e}")
              raise e
    # Catch RateLimitException specifically if needed
    except RateLimitException as rle:
         logger.error(f"VirusTotal Rate Limit hit fetching report for {url}: {rle}")
         raise rle # Re-raise
    except Exception as e:
         # Catch unexpected errors during the process
         logger.error(f"Unexpected error during VirusTotal check for {url}: {e}", exc_info=True)
         # Raising ApiException here informs the caller of failure
         raise ApiException(f"Unexpected error during VirusTotal check: {e}")

    # Fallback if logic has issues
    # return None


def parse_vt_result(vt_data: Optional[Dict]) -> Dict[str, Any]:
    """
    Parses the raw VirusTotal data object (expects the 'data' part of the full response).

    Returns a dictionary structured for the consolidation logic, including a 'status'.
    """
    if not vt_data or not isinstance(vt_data, dict) or 'attributes' not in vt_data:
        return {"status": "error", "details": "Invalid or missing VirusTotal data object passed to parser."}

    attributes = vt_data.get('attributes', {})
    if not attributes:
         return {"status": "error", "details": "Missing 'attributes' in VirusTotal data."}


    stats = attributes.get('last_analysis_stats')
    # Handle cases where stats are missing or empty (should have been caught by check_virustotal ideally)
    if not stats or not isinstance(stats, dict) or sum(stats.values()) == 0:
        logger.warning("Parsing VT result: last_analysis_stats missing or empty. Treating as incomplete.")
        return {"status": "pending", "details": "Analysis results not available or incomplete."}

    malicious_count = stats.get('malicious', 0)
    suspicious_count = stats.get('suspicious', 0)
    harmless_count = stats.get('harmless', 0)
    undetected_count = stats.get('undetected', 0)
    total_engines = sum(stats.values())

    # Determine assessment based on counts and threshold from config
    assessment = "unknown"
    if malicious_count >= MALICIOUS_THRESHOLD:
         assessment = "malicious"
    elif malicious_count > 0 or suspicious_count > 0:
         assessment = "suspicious"
    # Condition for 'likely_safe': primarily harmless/undetected votes
    elif (harmless_count + undetected_count) >= (total_engines * 0.9) and malicious_count == 0 and suspicious_count == 0 : # e.g. 90% safe/undetected
         assessment = "likely_safe"
    else: # Other combinations might be uncertain or need finer rules
         assessment = "uncertain"


    return {
        "status": "success", # Parsing successful, report contained results
        "details": {
            "assessment": assessment, # derived category
            "malicious_count": malicious_count,
            "suspicious_count": suspicious_count,
            "harmless_count": harmless_count,
            "undetected_count": undetected_count,
            "total_engines": total_engines,
            "positives": malicious_count + suspicious_count, # Often used metric
            "last_analysis_timestamp": attributes.get('last_analysis_date'), # Keep timestamp
            # Convert timestamp to readable string if needed here or in main logic
            # "last_analysis_date_human": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(attributes.get('last_analysis_date', 0))),
            "reputation": attributes.get("reputation", 0),
            "categories": attributes.get("categories", {}), # Dictionary of vendor categories
            "total_votes": attributes.get("total_votes", {}), # Community votes (harmless/malicious)
            "url_analysed": attributes.get("url", "N/A") # The URL VT actually analysed
        }
    }
