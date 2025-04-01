# api/utils.py
import re
from urllib.parse import urlparse, urljoin
import logging
import httpx # Import httpx
import asyncio

logger = logging.getLogger(__name__)

def is_valid_url(url: str) -> bool:
    """Validates if the given string looks like a URL."""
    try:
        # Handle potential missing schemes more robustly for validation
        if not url.startswith(('http://', 'https://')):
            # Attempt to parse with a default scheme for validation structure check
            parsed_temp = urlparse('http://' + url)
            return all([parsed_temp.netloc]) # Basic check: needs a domain part
        else:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
    except Exception as e:
        logger.warning(f"URL validation pattern matching failed: {e}")
        return False

async def ping_url(url: str, timeout: int = 5) -> bool:
    """
    Checks if a URL is reachable using an HTTP HEAD or GET request.
    Returns True if reachable (status 2xx or 3xx), False otherwise.
    """
    # Ensure URL has a scheme for the request
    if not url.startswith(('http://', 'https://')):
        # Default to http for pinging, but consider https might be needed
        url_http = 'http://' + url
        url_https = 'https://' + url
        urls_to_try = [url_https, url_http] # Prefer https
    else:
        urls_to_try = [url]

    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        for u in urls_to_try:
            try:
                # HEAD is lighter weight, but some servers don't support it well
                response = await client.head(u)
                # If HEAD fails or redirects strangely, try GET
                if 400 <= response.status_code < 500 or response.is_redirect:
                     response = await client.get(u) # follow_redirects handles this

                if 200 <= response.status_code < 400: # 2xx (OK) or 3xx (Redirects followed)
                    logger.info(f"URL {u} ping successful (Status: {response.status_code})")
                    return True
                else:
                    logger.warning(f"URL {u} ping failed (Status: {response.status_code})")
                    # Continue to next URL if multiple were generated

            except httpx.TimeoutException:
                logger.warning(f"URL {u} ping timed out after {timeout}s.")
                continue # Try next URL if available
            except httpx.RequestError as e:
                # Includes connection errors, invalid URL formats for httpx, etc.
                logger.warning(f"URL {u} ping failed: {type(e).__name__} - {e}")
                continue # Try next URL if available
            except Exception as e:
                logger.error(f"Unexpected error pinging {u}: {e}", exc_info=True)
                continue # Try next URL if available

    logger.warning(f"URL {url} could not be reached.")
    return False # Failed for all attempts

def sanitize_url_for_virustotal(url: str) -> str:
    """Ensures URL has a scheme, preferring https."""
    if not url.startswith(('http://', 'https://')):
        return 'https://' + url
    return url
