import logging
from typing import Optional, List, Dict, Any

import httpx
from dotenv import load_dotenv
import os
import asyncio

from .utils import get_config, RateLimitException, ApiException

load_dotenv()
CONFIG = get_config()
SEARCH_CONFIG = CONFIG['search_api']
API_KEY = os.getenv("SEARCH_API_KEY")
# SEARCH_API_PROVIDER = os.getenv("SEARCH_API_PROVIDER") # For reference
TIMEOUT = SEARCH_CONFIG['request_timeout']
RESULTS_COUNT = SEARCH_CONFIG['results_count']
ENGINE = SEARCH_CONFIG['engine'] # e.g., 'google'

# This URL is specific to SearchApi.io - change if using another service
BASE_API_URL = "https://www.searchapi.io/api/v1/search"

logger = logging.getLogger(__name__)

async def perform_search(query: str) -> Optional[List[Dict[str, str]]]:
    """
    Performs a web search using the configured Search API provider.

    Args:
        query: The search query string.

    Returns:
        A list of dictionaries, each containing 'title', 'link', 'snippet'
        for the search results, or None if an error occurs.
    Raises:
        RateLimitException: If the Search API indicates a rate limit issue.
        ApiException: For other client/server errors from the Search API.
    """
    if not API_KEY:
        logger.error("Search API key not found in environment variables.")
        return None

    params = {
        'api_key': API_KEY,
        'q': query,
        'engine': ENGINE,
        'num': str(RESULTS_COUNT) # API might expect string count
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            logger.debug(f"Performing web search via SearchApi.io for: {query}")
            response = await client.get(BASE_API_URL, params=params)

            # Check for rate limits (adjust based on provider's response)
            if response.status_code == 429: # Common rate limit code
                 logger.warning(f"Search API rate limit likely reached for query '{query}'. Status: {response.status_code}")
                 raise RateLimitException("Search API rate limit reached.")
            # Some APIs might use other codes or specific error messages in JSON
            response_json = response.json()
            if response.status_code >= 400 and "limit" in str(response_json).lower():
                 logger.warning(f"Search API rate limit suspected: Status {response.status_code}, Body: {response_json}")
                 raise RateLimitException("Search API rate limit suspected.")


            response.raise_for_status() # Check for other HTTP errors

            results = parse_search_results(response_json)
            logger.debug(f"Search for '{query}' yielded {len(results)} results.")
            return results

        except httpx.TimeoutException:
            logger.warning(f"Search API request timed out for query: {query}")
            return None
        except httpx.RequestError as req_err:
            logger.error(f"Network error querying Search API for '{query}': {req_err}")
            return None
        except httpx.HTTPStatusError as status_err:
            logger.error(f"Search API error for '{query}': Status {status_err.response.status_code}, Body: {status_err.response.text}")
            if status_err.response.status_code != 429:
                 raise ApiException(f"Search API error {status_err.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during web search for '{query}': {e}", exc_info=True)
            return None


def parse_search_results(search_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Parses the raw Search API response into a list of {'title', 'link', 'snippet'}.
    **NOTE: This structure is HIGHLY dependent on the API provider.**
    This is an example based on a common structure seen in Google-like results.
    """
    parsed_results = []
    # Adjust the keys based on YOUR Search API provider's response format!
    # Common keys might be 'organic_results', 'items', 'webPages', etc.
    organic_results = search_data.get('organic_results', [])

    for item in organic_results[:RESULTS_COUNT]: # Limit just in case API returns more
        title = item.get('title', 'No Title')
        link = item.get('link', '#')
        snippet = item.get('snippet', 'No Snippet')

        if title and link and snippet:
            parsed_results.append({
                "title": title.strip(),
                "link": link.strip(),
                "snippet": snippet.strip()
            })
    return parsed_results

# # Example usage (for testing)
# async def main_test():
#     test_query = "latest news on climate change policy"
#     results = await perform_search(test_query)
#     if results:
#         print(f"Search results for '{test_query}':")
#         for i, res in enumerate(results):
#             print(f"  {i+1}. {res['title']} ({res['link']})")
#             print(f"     Snippet: {res['snippet'][:100]}...")
#     else:
#         print(f"Failed to get search results for '{test_query}'")

# if __name__ == "__main__":
#     asyncio.run(main_test())
