# api/groq_utils.py

import logging
import httpx
import os
import json
import asyncio
import time # Keep for logging duration
import re
from typing import Optional, Dict, Any # Added Optional, Dict, Any
from functools import wraps # Keep wraps for decorator

from dotenv import load_dotenv

# --- Custom Exception Imports ---
# Make sure utils.py defines these exceptions
try:
    from .utils import get_config, RateLimitException, ApiException
except ImportError:
    # Fallback if run standalone for testing, though ideally it relies on the package structure
    print("Warning: Running groq_utils possibly standalone. Trying relative path for utils.")
    from utils import get_config, RateLimitException, ApiException # type: ignore


# Specific Exception for Groq errors
class GroqApiException(ApiException):
    """Custom exception specifically for Groq API errors."""
    pass

# --- Setup ---
load_dotenv()
logger = logging.getLogger(__name__) # Use standard logger name
CONFIG = get_config() # Use central config getter
GROQ_CONFIG = CONFIG.get('groq', {}) # Use .get for safety
API_KEY = os.getenv("GROQ_API_KEY")
REQUEST_TIMEOUT = GROQ_CONFIG.get('request_timeout', 45) # Get timeout from config

# Shared HTTP client managed by main.py's lifespan manager
_groq_client: Optional[httpx.AsyncClient] = None

def setup_groq_client():
    """Initializes the shared httpx client for Groq."""
    global _groq_client
    # Check if already initialized (idempotent)
    if _groq_client is not None:
        return

    if not API_KEY:
        logger.error("GROQ_API_KEY not found in environment variables. Groq queries will fail.")
        # Service can run without Groq, but log error clearly
        _groq_client = None # Explicitly set to None
        return

    try:
        _groq_client = httpx.AsyncClient(
            base_url="https://api.groq.com/openai/v1",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=REQUEST_TIMEOUT + 10, # Client timeout slightly higher than request timeout
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )
        logger.info("Shared Groq HTTP client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize shared Groq HTTP client: {e}", exc_info=True)
        _groq_client = None

async def close_groq_client():
    """Closes the shared httpx client."""
    global _groq_client
    if _groq_client:
        try:
            await _groq_client.aclose()
            logger.info("Shared Groq HTTP client closed.")
        except Exception as e:
            logger.error(f"Error closing shared Groq HTTP client: {e}", exc_info=True)
        finally:
             _groq_client = None # Ensure it's marked as closed/None

# --- Main Groq Query Function ---
async def query_groq(
    prompt: str,
    model: str = None, # Use default from config if None
    temperature: float = None, # Use default from config if None
    max_tokens: int = 2048 # Default max_tokens, adjust if needed
    ) -> Optional[str]:
    """
    Sends a query to the Groq API using the shared client and returns the content.

    Raises:
        RateLimitException: If Groq API returns 429.
        GroqApiException: For other 4xx/5xx errors from Groq or network issues.
        ValueError: If essential configuration (model, client) is missing.
    """
    global _groq_client
    if _groq_client is None:
         # It should be initialized by the lifespan manager before this is called
         logger.error("Groq client is not initialized. Cannot query API.")
         # Raise an exception because the application state is wrong
         raise GroqApiException("Groq client has not been initialized. Check application startup.")

    # Use defaults from config if arguments are None
    model_to_use = model or GROQ_CONFIG.get('model')
    temp_to_use = temperature if temperature is not None else GROQ_CONFIG.get('temperature')

    if not model_to_use:
         logger.error("Groq model name is not specified in call or config.")
         raise ValueError("Missing Groq model configuration.")
    if temp_to_use is None: # Should have a default in config, but double check
        logger.warning("Groq temperature not specified, using default 0.1")
        temp_to_use = 0.1

    payload = {
        "model": model_to_use,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp_to_use,
        "max_tokens": max_tokens,
    }

    try:
        request_start_time = time.monotonic()
        logger.debug(f"Sending request to Groq model {model_to_use}. Prompt: '{prompt[:100]}...'")

        response = await _groq_client.post(
            "/chat/completions",
            json=payload,
            timeout=REQUEST_TIMEOUT # Per-request timeout from config
        )

        request_duration = time.monotonic() - request_start_time
        logger.debug(f"Groq response received in {request_duration:.3f}s. Status: {response.status_code}")

        # --- Specific Error Handling ---
        if response.status_code == 429:
            logger.warning(f"Groq API rate limit hit (Status 429). Prompt: '{prompt[:50]}...'")
            raise RateLimitException("Groq API rate limit exceeded.")

        # Handle other client/server errors
        if response.status_code >= 400:
            error_message = f"Groq API error {response.status_code}"
            try:
                error_details = response.json()
                error_message += f": {error_details}"
                logger.error(f"Groq API Error Response: {error_details}")
            except json.JSONDecodeError:
                error_message += f": {response.text}"
                logger.error(f"Groq API Error Response (non-JSON): {response.text}")
            raise GroqApiException(error_message)
        # --- End Specific Error Handling ---

        # If success (2xx)
        data = response.json()
        choices = data.get("choices")
        if choices and isinstance(choices, list) and len(choices) > 0:
            message = choices[0].get("message")
            if message and isinstance(message, dict):
                 content = message.get("content")
                 if content:
                      logger.debug(f"Groq ({model_to_use}) response content: {content[:100]}...")
                      return content.strip()

        # Handle unexpected successful response structure
        logger.warning(f"Groq response structure unexpected or content missing. Full response: {data}")
        return None # Return None if content parsing fails

    # Handle client-side or network errors
    except httpx.TimeoutException:
        logger.warning(f"Groq request timed out after {REQUEST_TIMEOUT}s. Prompt: '{prompt[:50]}...'")
        raise GroqApiException(f"Groq API request timed out ({REQUEST_TIMEOUT}s).")
    except httpx.RequestError as req_err:
        # E.g., DNS resolution error, connection refused, etc.
        logger.error(f"Network error connecting to Groq: {req_err}")
        raise GroqApiException(f"Network error contacting Groq: {req_err}")
    # Re-raise specific custom exceptions if they somehow occurred before this block
    except (RateLimitException, GroqApiException) as api_exc:
         raise api_exc
    # Catch any other unexpected exceptions
    except Exception as e:
        logger.error(f"Unexpected error during Groq query: {e}", exc_info=True)
        raise GroqApiException(f"Unexpected error during Groq communication: {e}")


# --- Higher-Level Groq Functions ---
# (analyze_misinformation_groq, ask_groq_factual, extract_intent_groq)
# These functions remain structurally similar, but now they call the updated
# `query_groq` function which handles client management and error raising.
# We need to ensure their `try...except` blocks correctly handle the
# `RateLimitException` and `GroqApiException` that `query_groq` might raise.

async def analyze_misinformation_groq(text: str, model: str = None) -> Dict[str, Any]:
    """Analyze text for misinformation using Groq ONLY."""
    logger.debug(f"Initiating Groq-only misinformation analysis for: '{text[:100]}...'")
    try:
        # Prompt assumes the detailed instructions from previous versions
        prompt = f"""
        ## Task: Analyze the following statement for misinformation
        Statement: "{text}"
        ## Analysis Instructions: Provide a critical assessment focusing on accuracy, potential bias, logical fallacies, and emotional manipulation. Avoid definitive "true/false" unless widely accepted consensus exists. State limitations (e.g., knowledge cutoff).
        ## Classification System: (Use ONE Category)
        *   **likely_factual:** Content appears accurate based on common knowledge.
        *   **likely_misleading:** Contains factual inaccuracies, distortions, or omits critical context.
        *   **opinion:** Primarily subjective view, not presented as objective fact.
        *   **satire:** Humorous or exaggerated content not intended as factual. (Use cautiously)
        *   **needs_verification:** Insufficient information in statement or common knowledge to assess.
        *   **contradictory:** Reliable information sources conflict on this topic.
        *   **other:** Doesn't fit other categories (e.g., question, command).
        ## Confidence Score: Provide a score from 0.0 (no confidence) to 1.0 (high confidence) in your classification.
        ## Response Format: STRICTLY return ONLY a single JSON object enclosed in ```json ... ``` with the following keys: "category" (string), "confidence" (float), "explanation" (string, detailed reasoning), "recommendation" (string, advice for user), "key_issues" (list of strings, identified potential problems like 'unsourced claim', 'emotional language', etc.), "verifiable_claims" (list of strings, specific factual claims that *could* be independently verified, if any).
        """

        # No temperature override here, rely on default low temp for analysis
        raw_response_content = await query_groq(prompt, model=model) # Error handling within query_groq

        if not raw_response_content:
            raise GroqApiException("Groq returned empty content for misinformation analysis.")

        # --- JSON Parsing (robust logic from before is good) ---
        json_result = None
        try:
            json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', raw_response_content, re.IGNORECASE)
            if json_match:
                json_str = json_match.group(1)
                json_result = json.loads(json_str)
            else:
                 # Fallback: Try finding first '{' and last '}'
                json_start = raw_response_content.find("{")
                json_end = raw_response_content.rfind("}") + 1
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    json_str = raw_response_content[json_start:json_end]
                    json_result = json.loads(json_str)
                else:
                    logger.warning("Could not extract JSON block from Groq misinfo response.")
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse JSON from Groq misinfo response: {json_err}. Response: '{raw_response_content[:200]}...'")
            # Fall through, will create default error below if json_result is None

        if json_result and isinstance(json_result, dict):
            # Validate/normalize expected fields
            required_fields = ["category", "confidence", "explanation", "recommendation"]
            for field in required_fields: json_result.setdefault(field, "Not Provided")
            json_result.setdefault("key_issues", [])
            json_result.setdefault("verifiable_claims", [])

            # Optional: Validate category against allowed values if strictness needed
            allowed_cats = {"likely_factual", "likely_misleading", "opinion", "satire", "needs_verification", "contradictory", "other"}
            if json_result.get("category") not in allowed_cats:
                logger.warning(f"Groq returned unexpected category '{json_result.get('category')}'. Check prompt/model behavior.")
                # Optionally map it to 'other' or leave as is

            logger.info(f"Groq Misinfo analysis complete. Category: {json_result.get('category', 'unknown')}")
            return json_result
        else:
            # Parsing failed or result wasn't a dict
            return {
                "category": "error", "confidence": 0.0,
                "explanation": "Failed to parse valid JSON response from LLM.",
                "recommendation": "Please try rephrasing or contact support.",
                "key_issues": [], "verifiable_claims": [],
                "raw_response": raw_response_content # Include raw response for debugging
            }

    except (RateLimitException, GroqApiException) as api_err:
         logger.error(f"Groq API error during misinfo analysis: {api_err}")
         return {
             "category": "error", "confidence": 0.0,
             "explanation": f"Analysis failed due to API error: {api_err}",
             "recommendation": "Try again later.", "key_issues": [], "verifiable_claims": []
         }
    except Exception as e:
         logger.error(f"Unexpected error during misinfo analysis: {e}", exc_info=True)
         return {
             "category": "error", "confidence": 0.0,
             "explanation": f"Unexpected analysis error: {e}",
             "recommendation": "Try again or contact support.", "key_issues": [], "verifiable_claims": []
         }

async def ask_groq_factual(question: str, model: str = None) -> Dict[str, Any]:
    """Handle factual QA using Groq ONLY."""
    logger.debug(f"Initiating Groq-only factual QA for: '{question[:100]}...'")
    try:
        # Simple, direct factual prompt
        prompt = f"Please answer the following question factually and concisely. If you don't know the answer or it requires knowledge beyond your cutoff date, please state that clearly.\n\nQuestion: \"{question}\"\n\nAnswer:"

        # Use low temperature for factual consistency
        answer_content = await query_groq(prompt, model=model, temperature=0.05)

        if not answer_content:
             raise GroqApiException("Groq returned empty content for factual question.")

        # Basic confidence assessment based on response content
        confidence = "high"
        lower_content = answer_content.lower()
        uncertainty_phrases = ["i don't know", "i'm not sure", "beyond my knowledge cutoff", "unable to provide", "no information", "cannot confirm", "as an ai", "my knowledge is limited"]
        if any(phrase in lower_content for phrase in uncertainty_phrases):
             confidence = "low" # Changed to low if uncertainty explicit
        elif any(phrase in lower_content for phrase in ["possibly", "might be", "could be", "likely", "probably"]):
              confidence = "medium" # Medium if speculative

        result = {
            "question": question,
            "answer": answer_content,
            "confidence_level": confidence,
            "source": "Groq Internal Knowledge",
            "model_used": model or GROQ_CONFIG.get('model')
        }
        logger.info(f"Groq Factual question answered. Confidence: {confidence}")
        return result

    except (RateLimitException, GroqApiException) as api_err:
        logger.error(f"Groq API error during factual QA: {api_err}")
        return {
            "question": question, "answer": f"Failed to get answer due to API error: {api_err}",
            "confidence_level": "error", "source": "API Error", "model_used": model
        }
    except Exception as e:
        logger.error(f"Unexpected error during factual QA: {e}", exc_info=True)
        return {
             "question": question, "answer": f"Unexpected error during question answering: {e}",
            "confidence_level": "error", "source": "System Error", "model_used": model
         }

# --- extract_intent_groq: Keeping but marking as secondary ---
async def extract_intent_groq(query: str, model: str = None) -> str:
    """Determine the primary intent using Groq (Secondary/Fallback)."""
    logger.debug("Attempting Groq-based intent classification (secondary)...")
    # Check if local classifier is primary method elsewhere and maybe skip this?
    # Assumes this is called specifically when fallback is needed.
    try:
        # Keep the prompt simple for classification
        prompt = f"""Classify the user's primary intent from the following query into one of these categories: 'misinfo_check', 'url_analysis', 'factual_question', 'other'. Query: "{query}" Intent: """
        # Use a cheaper/faster model maybe? e.g. llama3-8b
        intent_response = await query_groq(prompt, model=(model or "llama3-8b-8192"), temperature=0.0)

        if not intent_response:
            logger.warning("Groq intent classification returned empty content.")
            return "other"

        intent = intent_response.strip().lower().replace("'", "").replace("\"", "")
        # Simple mapping
        if "misinfo_check" in intent or "check_fact" in intent: return "misinfo"
        if "url_analysis" in intent or "url_safety" in intent: return "url"
        if "factual_question" in intent or "general_question" in intent: return "factual"

        logger.warning(f"Groq returned ambiguous intent: '{intent}'. Classifying as 'other'.")
        return "other"
    except Exception as e:
        # Catch API errors or unexpected issues
        logger.error(f"Error during Groq intent classification: {e}", exc_info=True)
        return "other" # Default on error
