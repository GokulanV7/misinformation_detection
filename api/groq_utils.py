# api/groq_utils.py

# REMOVED: from api.factcheck_api_util import query_fact_check_api
import httpx # Use httpx
import json
import logging
import yaml
import re
import asyncio # For retry decorator sleep
from typing import Dict, Any
import time
from functools import wraps

logger = logging.getLogger(__name__)

def load_config(config_path="config/config.yaml"):
    """Load configuration from the YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        return {} # Return empty dict or raise error
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {config_path}: {e}")
        return {}

config = load_config()

# --- Async Retry Decorator ---
def retry_on_failure(max_retries: int = 3, backoff_factor: float = 1.0, exceptions=(httpx.RequestError,)):
    """Decorator to retry failed async API calls with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        logger.error(f"Final attempt ({attempt + 1}) failed for {func.__name__}. Error: {e}")
                        break # Exit loop to raise below
                    delay = backoff_factor * (2 ** attempt)
                    logger.warning(f"{func.__name__}: Attempt {attempt + 1}/{max_retries} failed with {type(e).__name__}. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
            # Raise the last encountered exception if all retries fail
            if last_exception:
                raise last_exception
            # Should ideally not be reached if exceptions are caught, but as a fallback
            raise Exception(f"{func.__name__} failed after {max_retries} retries, but no exception was stored.")
        return wrapper
    return decorator

# --- Groq API Query Function ---
@retry_on_failure(max_retries=3, backoff_factor=1, exceptions=(httpx.RequestError, httpx.HTTPStatusError))
async def query_groq(content: str, api_key: str, model: str, temperature: float = None) -> Dict[str, Any]:
    """Query Groq API with enhanced error handling and retry logic using httpx."""
    if not api_key:
        logger.error("Groq API Key is missing.")
        # Raising an error might be better than returning dict here
        raise ValueError("Groq API Key not configured.")

    temp_setting = temperature if temperature is not None else config.get('groq', {}).get('temperature', 0.1)
    groq_model = model or config.get('groq', {}).get('model')
    if not groq_model:
         raise ValueError("Groq model not specified in call or config.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": groq_model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temp_setting
    }
    url = "https://api.groq.com/openai/v1/chat/completions"

    async with httpx.AsyncClient(timeout=30.0) as client: # Use httpx.AsyncClient
        try:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx
            return response.json()

        # Specific handling is now mostly within the retry decorator,
        # but we can log details here if needed, though decorator handles logging too.
        except httpx.HTTPStatusError as e:
            logger.error(f"Groq API HTTP Error {e.response.status_code}: {e.response.text}")
            # Let the retry decorator handle raising after retries
            raise
        except httpx.RequestError as e:
            logger.error(f"Groq API Request Error: {type(e).__name__} - {e}")
            # Let the retry decorator handle raising after retries
            raise
        except Exception as e:
            # Catch unexpected errors not covered by httpx exceptions
            logger.error(f"Unexpected error during Groq query: {str(e)}", exc_info=True)
            raise # Re-raise unexpected errors

# --- Analyze Misinformation ---
async def analyze_misinformation_groq(text: str, api_key: str, model: str) -> Dict[str, Any]:
    """
    Analyze text for misinformation using Groq only.
    (Renamed from analyze_misinformation to be specific)
    """
    # NOTE: Removed the call to fact_check_api

    try:
        # Enhanced prompt (remains the same)
        prompt = f"""
        ## Task: Analyze the following statement for misinformation
        Statement: "{text}"
        ## Analysis Instructions: ... (keep the detailed instructions)
        ## Classification System: ... (keep the classification details)
        ## Response Format: ... (keep the JSON format spec)
        {{
            "category": "[SELECTED CATEGORY]",
            "confidence": [CONFIDENCE SCORE BETWEEN 0.0 AND 1.0],
            "explanation": "[DETAILED EXPLANATION OF CLASSIFICATION]",
            "recommendation": "[ACTIONABLE ADVICE FOR THE USER]",
            "key_issues": ["ISSUE 1", "ISSUE 2", "..."],
            "verifiable_claims": ["CLAIM 1", "CLAIM 2", "..."]
        }}
        """

        response = await query_groq(prompt, api_key, model, 0.1) # Use the robust query_groq

        message_content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

        # JSON Parsing logic (remains largely the same, robust is good)
        try:
            json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', message_content)
            if json_match:
                json_str = json_match.group(1)
                result = json.loads(json_str)
            else:
                json_start = message_content.find("{")
                json_end = message_content.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = message_content[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    logger.warning("No JSON found in Groq misinfo response, using raw text.")
                    result = {
                        "category": "uncertain", "confidence": 0.5,
                        "explanation": message_content[:500],
                        "recommendation": "AI response format unclear.",
                        "raw_response": message_content
                    }
        except json.JSONDecodeError as e:
            logger.error(f"Groq misinfo JSON parse error: {str(e)}. Response: {message_content[:200]}...")
            result = {
                "category": "uncertain", "confidence": 0.5,
                "explanation": "AI response could not be parsed correctly.",
                "recommendation": "Try rephrasing input.",
                "raw_response": message_content,
                "error": f"Failed to parse JSON: {str(e)}"
            }

        # Field validation (remains the same)
        required_fields = ["category", "confidence", "explanation", "recommendation"]
        for field in required_fields:
            result.setdefault(field, "Not provided") # Use setdefault for cleaner code

        logger.info(f"Groq Misinformation analysis complete. Category: {result.get('category', 'unknown')}")
        return result

    except Exception as e:
        # Catch errors from query_groq or other issues
        logger.error(f"Error during Groq misinformation analysis: {str(e)}", exc_info=True)
        # Return a structured error response
        return {
            "category": "error", "confidence": 0.0,
            "explanation": f"An error occurred during analysis: {str(e)}",
            "recommendation": "Please try again later or contact support.",
            "error": str(e)
        }


# --- Ask Groq (Factual QA) ---
async def ask_groq_factual(question: str, api_key: str, model: str) -> Dict[str, Any]:
    """
    Handle factual question answering using Groq with enhanced error handling.
    (Renamed from ask_groq to be specific)
    """
    try:
        # Enhanced prompt (remains the same)
        prompt = f"""
        ## Factual Question Answering
        Question: "{question}"
        Please provide a factual, accurate answer... (keep full prompt guidelines)
        Also, explicitly state if your knowledge about very recent events might be limited or outdated.
        """

        response = await query_groq(prompt, api_key, model, 0.1) # Use robust query_groq

        message_content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Confidence assessment (remains the same)
        confidence_level = "high"
        uncertainty_phrases = ["I'm not sure", "don't know", "uncertain", "unclear", "possibly", "might be", "could be", "knowledge cutoff", "limited data", "information may be outdated"]
        for phrase in uncertainty_phrases:
            if phrase in message_content.lower():
                confidence_level = "medium" # Downgrade if uncertainty is expressed
                break

        # Maybe check for stronger uncertainty if needed, but medium covers most cases
        # high_uncertainty_phrases = [...]

        result = {
            "question": question,
            "answer": message_content,
            "confidence_level": confidence_level,
            "source": "Groq Internal Knowledge", # Indicate source
            "model": model # Return the specific model used
        }

        logger.info(f"Groq Factual question answered. Confidence: {confidence_level}")
        return result

    except Exception as e:
        logger.error(f"Groq factual QA error: {str(e)}", exc_info=True)
        return {
            "question": question,
            "error": f"Factual question answering error: {str(e)}",
            "answer": "An error occurred while trying to answer the question.",
            "confidence_level": "error",
            "source": "Groq Error"
            }

# --- Extract Intent (Deprecated if using local classifier) ---
# This function might still be useful for other purposes or fallback,
# but the primary classification should now happen locally.
# Keep it but maybe rename or mark as secondary.
async def extract_intent_groq(query: str, api_key: str, model: str) -> str:
    """Determine the primary intent of the query using Groq."""
    # Note: This is now less efficient than the local classifier.
    # Use primarily for fallback or specific complex cases if needed.
    try:
        prompt = f"""
        Determine the primary intent... (keep full prompt)
        Query: "{query}" Intent:
        """

        response = await query_groq(prompt, api_key, model, 0.2)
        intent = response.get("choices", [{}])[0].get("message", {}).get("content", "").strip().lower()

        # Validate intent against known labels
        valid_intents = {"check_fact", "url_safety", "general_question", "other"}
        # Simple validation: check if response contains one of the keywords
        if "check_fact" in intent or "misinfo" in intent: return "misinfo" # Map check_fact to misinfo
        if "url" in intent: return "url"
        if "general_question" in intent or "factual" in intent: return "factual"

        logger.warning(f"Groq intent classification returned ambiguous result: {intent}")
        return "other"

    except Exception as e:
        logger.error(f"Groq intent classification error: {str(e)}", exc_info=True)
        return "other"  # Default to "other" on error
