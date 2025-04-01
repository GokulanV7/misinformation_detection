# api/vt_utils.py
import httpx # Use httpx
import base64
import logging
import yaml
import asyncio # For potential sleep
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_config(config_path="config/config.yaml"):
    """Load configuration from the YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
# REMOVED hardcoded placeholder: VIRUSTOTAL_API_KEY = ""

# Define VirusTotal API endpoint
VIRUSTOTAL_API_URL = config['virustotal']['api_url'] # Use config

async def check_url_safety(url: str, api_key: str, timeout: int = 20) -> Dict[str, Any]:
    """
    Checks URL safety using VirusTotal API asynchronously.
    Handles getting existing reports or submitting new URLs.
    """
    if not api_key:
        logger.error("VirusTotal API Key is missing.")
        return {"error": "VirusTotal API Key not configured."}

    headers = {
        "x-apikey": api_key,
        "accept": "application/json" # Ensure we get JSON back
    }
    url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
    report_url = f"{VIRUSTOTAL_API_URL}/{url_id}"

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            # 1. Try to get an existing report
            logger.info(f"Checking VirusTotal for existing report: {url}")
            response = await client.get(report_url, headers=headers)

            if response.status_code == 200:
                logger.info(f"Found existing VirusTotal report for URL: {url}")
                result = response.json()
                return _parse_vt_result(result, url) # Parse the result directly

            elif response.status_code == 404:
                # 2. URL not found, submit it for analysis
                logger.info(f"URL not found in VirusTotal, submitting for analysis: {url}")
                submit_data = {"url": url}
                # VirusTotal submission expects 'application/x-www-form-urlencoded'
                headers_submit = headers.copy()
                headers_submit['content-type'] = 'application/x-www-form-urlencoded'
                submit_response = await client.post(VIRUSTOTAL_API_URL, headers=headers_submit, data=submit_data)

                if submit_response.status_code == 200:
                    submit_result = submit_response.json()
                    analysis_id = submit_result.get("data", {}).get("id")
                    if not analysis_id:
                         logger.error("Failed to get analysis ID after submission.")
                         return {"error": "VirusTotal submission succeeded but no analysis ID returned."}

                    logger.info(f"URL submitted successfully. Analysis ID: {analysis_id}. Waiting for report...")
                    # Optionally: Poll the analysis endpoint - this can take time!
                    # For a simple API, returning a "pending" status might be better
                    # than blocking the request here for potentially minutes.
                    # For now, we'll just indicate it's submitted.
                    # A more robust solution would use webhooks or background tasks.
                    await asyncio.sleep(10) # Give VT some time (very basic polling)
                    poll_url = f"https://www.virustotal.com/api/v3/analyses/{analysis_id}"
                    poll_response = await client.get(poll_url, headers=headers)
                    if poll_response.status_code == 200:
                        poll_result = poll_response.json()
                        # The actual results are linked, need to fetch the final URL report again
                        final_report_url = poll_result.get("meta", {}).get("url_info", {}).get("links",{}).get("self")
                        if final_report_url:
                             final_response = await client.get(final_report_url, headers=headers)
                             if final_response.status_code == 200:
                                 return _parse_vt_result(final_response.json(), url)

                    return {"status": "submitted", "message": "URL submitted to VirusTotal for analysis. Check back later."}


                else:
                    logger.error(f"VirusTotal submission failed. Status: {submit_response.status_code}, Response: {submit_response.text}")
                    return {"error": f"VirusTotal submission failed (Status: {submit_response.status_code})", "details": submit_response.text}
            else:
                # Handle other unexpected status codes
                response.raise_for_status() # Raise error for other non-200, non-404 codes

        except httpx.HTTPStatusError as e:
             logger.error(f"VirusTotal API HTTP Error {e.response.status_code}: {e.response.text} for URL {url}")
             return {"error": f"VirusTotal API HTTP Error {e.response.status_code}", "details": e.response.text}
        except httpx.RequestError as e:
             logger.error(f"VirusTotal API Request Error: {e} for URL {url}")
             return {"error": f"VirusTotal API connection error: {str(e)}"}
        except Exception as e:
             logger.error(f"Unexpected error processing VirusTotal URL analysis: {e} for URL {url}", exc_info=True)
             return {"error": f"Unexpected error during VirusTotal analysis: {str(e)}"}

    # Should not be reached if using async with, but added for safety
    return {"error": "Failed to process VirusTotal request."}


def _parse_vt_result(result: Dict[str, Any], original_url: str) -> Dict[str, Any]:
    """Helper function to parse the JSON response from VirusTotal API."""
    attributes = result.get("data", {}).get("attributes", {})
    if not attributes:
        return {"url": original_url, "status": "error", "message": "No attributes found in VirusTotal response."}

    last_analysis_stats = attributes.get("last_analysis_stats", {})
    if not last_analysis_stats:
         # Can happen if analysis is queued or just finished with no engine results yet
         return {"url": original_url, "status": "pending", "message": "VirusTotal analysis is pending or has no results yet."}

    total_engines = sum(last_analysis_stats.values()) or 1
    harmless_count = last_analysis_stats.get("harmless", 0)
    malicious_count = last_analysis_stats.get("malicious", 0)
    suspicious_count = last_analysis_stats.get("suspicious", 0)

    safety_score = (harmless_count / total_engines) * 100
    danger_score = ((malicious_count + suspicious_count) / total_engines) * 100

    # Get assessment and safety category from configuration
    default_assessment = config['virustotal']['default_assessment']
    assessment = default_assessment

    if malicious_count > 0:
        assessment = "Malicious"
    elif suspicious_count > 0:
        assessment = "Suspicious"
    # Add specific check for potentially unwanted applications (PUA) if needed
    # elif last_analysis_stats.get("undetected", 0) < total_engines * 0.1: # Example heuristic
    #     assessment = "Potentially Unwanted"

    simplified_result = {
        "url": attributes.get("url", original_url), # Use URL from VT if available
        "status": "completed", # Indicate the scan itself finished
        "scan_stats": last_analysis_stats,
        "last_analysis_date": attributes.get("last_analysis_date"),
        "reputation": attributes.get("reputation", 0),
        "title": attributes.get("title", "N/A"),
        "assessment": assessment,
        "safety_score": round(safety_score, 1),
        "danger_score": round(danger_score, 1),
        "categories": attributes.get("categories", {}),
        "total_votes": attributes.get("total_votes", {})
    }

    logger.info(f"VirusTotal analysis complete for {original_url}. Assessment: {assessment}")
    return simplified_result
