# api/urlscan_utils.py
import logging
from typing import Optional, Dict, Any
import httpx
from dotenv import load_dotenv
import os
import json # <-- Ensure imported for parsing safety

# --- Custom Exception Imports ---
try:
    from .utils import get_config, RateLimitException, ApiException
except ImportError:
    print("Warning: Running urlscan_utils possibly standalone.")
    from utils import get_config, RateLimitException, ApiException # type: ignore

# --- Setup ---
load_dotenv()
CONFIG = get_config()
URLSCAN_CONFIG = CONFIG.get('urlscan', {}) # Use .get()
API_KEY = os.getenv("URLSCAN_API_KEY")
SEARCH_API_URL = URLSCAN_CONFIG.get('search_api_url', "https://urlscan.io/api/v1/search/")
TIMEOUT = URLSCAN_CONFIG.get('request_timeout', 25)
logger = logging.getLogger(__name__)

# --- Main Check Function ---
async def check_urlscan_existing_results(url: str) -> Optional[Dict[str, Any]]:
    """
    Searches urlscan.io for existing results based on the domain of the URL.

    Returns: Raw result object from urlscan API on success (latest result), None if no results found,
             or on timeout/network errors.
    Raises: RateLimitException, ApiException on specific API errors.
    """
    if not url:
         logger.warning("URLScan check called with empty URL.")
         return None

    headers = {'Content-Type': 'application/json'}
    if API_KEY: headers['API-Key'] = API_KEY # Add key if present

    try:
        # Extract domain for searching - use httpx URL parsing
        parsed_url = httpx.URL(url)
        domain = parsed_url.host
        if not domain: raise ValueError("Could not parse domain from URL.")
    except Exception as parse_err:
        logger.error(f"Failed to parse domain from URL '{url}': {parse_err}")
        raise ValueError(f"Invalid URL provided for URLScan check: {url}") from parse_err

    query = f"domain:{domain}"
    params = {'q': query, 'size': '1'} # Get only the latest result by default

    async with httpx.AsyncClient(timeout=TIMEOUT, headers=headers) as client:
        request_start_time = time.monotonic()
        logger.debug(f"Sending request to URLScan Search: Query='{query}'")
        try:
            response = await client.get(SEARCH_API_URL, params=params)
            request_duration = time.monotonic() - request_start_time
            logger.debug(f"URLScan response received in {request_duration:.3f}s. Status: {response.status_code}")

            # --- Error Checks ---
            if response.status_code == 429:
                logger.warning(f"URLScan rate limit (429) for query '{query}'.")
                raise RateLimitException("URLScan API rate limit reached.")
            if response.status_code == 400: # Often bad query syntax
                try: detail=response.json()
                except: detail=response.text
                logger.error(f"URLScan Bad Request (400) for query '{query}'. Response: {detail}")
                raise ApiException(f"URLScan API Bad Request (check query format): {detail}")
            if response.status_code == 401:
                logger.error(f"URLScan Unauthorized (401) for query '{query}'. Check API Key if required for search.")
                raise ApiException("URLScan API Key invalid or required for search.")
            if response.status_code >= 500: # Server errors
                logger.error(f"URLScan server error ({response.status_code}) for query '{query}'. Response: {response.text}")
                raise ApiException(f"URLScan server error {response.status_code}.")
            # Check other specific codes if necessary
            response.raise_for_status() # Catch any remaining 4xx not explicitly handled
            # --- End Error Checks ---

            # Parse JSON response
            data = response.json()

            # Check structure and get results
            if data and isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
                results_list = data["results"]
                if len(results_list) > 0:
                    latest_result = results_list[0] # Get the first (should be latest)
                    logger.info(f"Found existing URLScan result for domain '{domain}'. Scan ID: {latest_result.get('task',{}).get('uuid')}")
                    # Return the raw result object for the parser function
                    return latest_result
                else:
                    logger.info(f"No existing URLScan results found for domain: {domain}")
                    return None # Explicitly None if no results array or empty
            else:
                logger.warning(f"Unexpected response structure from URLScan search for '{query}'. Data: {str(data)[:200]}")
                return None # Return None if structure is wrong

        except httpx.TimeoutException:
            logger.warning(f"URLScan search request timed out for query: '{query}'")
            return None
        except httpx.RequestError as req_err:
            logger.error(f"Network error contacting URLScan search for '{query}': {req_err}")
            return None
        except json.JSONDecodeError as json_err:
             logger.error(f"URLScan returned non-JSON response ({response.status_code}). Body: '{response.text[:200]}...'. Error: {json_err}")
             return None
        # Re-raise specific API exceptions
        except (RateLimitException, ApiException, ValueError) as e:
            raise e
        # Catch any other unexpected errors
        except Exception as e:
            logger.error(f"Unexpected error during URLScan search for '{query}': {e}", exc_info=True)
            raise ApiException(f"Unexpected error during URLScan search: {e}") from e


def parse_urlscan_result(scan_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parses the raw urlscan.io *search result item* object.
    Returns a dictionary structured for consolidation logic.
    """
    if not scan_data or not isinstance(scan_data, dict):
        # If called with None (e.g., no scan found), return specific status
        return {"status": "no_scan_found", "details": {"message": "No existing scan found for domain."}}

    details = {}
    status = "scan_found" # Assume found if data is provided
    assessment = "benign" # Default assessment

    try:
        # Safely extract nested data using .get() with default {} or []
        task_info = scan_data.get("task", {})
        page_info = scan_data.get("page", {})
        verdicts = scan_data.get("verdicts", {})
        # URLScan has multiple verdict blocks, 'overall' is a good summary
        overall_verdict = verdicts.get("overall", {})
        # urlscan_verdict = verdicts.get("urlscan", {}) # Specific urlscan verdict if needed

        # --- Populate Details ---
        details["scan_id"] = task_info.get("uuid")
        details["scan_date"] = task_info.get("time") # ISO 8601 format
        details["scanned_url"] = page_info.get("url") # Actual URL scanned in this task
        details["effective_url"] = page_info.get("effectiveUrl", details["scanned_url"]) # URL after redirects
        details["domain"] = page_info.get("domain")
        details["ip_address"] = page_info.get("ip")
        details["asn_name"] = page_info.get("asnname")
        details["server_country"] = page_info.get("country")

        # Verdict Info
        details["score"] = overall_verdict.get("score", 0)
        details["verdict_malicious"] = overall_verdict.get("malicious", False) # Overall malicious flag
        details["verdict_has_ads"] = overall_verdict.get("has_ads", False)
        tags = overall_verdict.get("tags", [])
        details["tags"] = tags # List of strings
        details["categories"] = overall_verdict.get("categories", [])

        details["report_url"] = scan_data.get("result") # Link to full report page
        details["screenshot_url"] = scan_data.get("screenshot")

        # --- Derive Assessment ---
        # More nuanced assessment based on multiple factors
        score = details["score"]
        is_malicious_verdict = details["verdict_malicious"]
        suspicious_tags = ['phishing', 'malware', 'suspicious', 'hacked', 'crypto', 'spam', 'dynamic dns', 'redirector']

        if is_malicious_verdict:
             assessment = "malicious"
        elif any(tag in suspicious_tags for tag in tags):
             assessment = "suspicious"
        elif score > 75: # High score is suspicious even without specific tags/verdict
             assessment = "suspicious"
        # Keep 'benign' as the default if none of the above trigger

        details["assessment_category"] = assessment
        # --- End Assessment ---

        # Filter None values for cleaner output if desired
        filtered_details = {k: v for k, v in details.items() if v is not None}
        return {"status": status, "details": filtered_details}

    except Exception as e:
         logger.error(f"Error parsing URLScan result structure: {e}. Data: {str(scan_data)[:200]}...", exc_info=True)
         return {"status": "error", "details": {"message": f"Failed to parse scan data: {e}"}}
