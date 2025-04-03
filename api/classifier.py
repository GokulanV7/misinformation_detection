# api/classifier.py
import logging
# REMOVED: from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
# Use only the pipeline for zero-shot
from transformers import pipeline
import torch
import os
from typing import Tuple


# Ensure utils can be imported for config
try:
    from .utils import get_config
except ImportError:
    # Fallback for potential standalone testing
    print("Warning: Could not import get_config from .utils. Configuration might be hardcoded.")
    # Define a dummy get_config or hardcode values if testing standalone
    def get_config(): return {
        "classifier": {
            "model_name": "facebook/bart-large-mnli", # Default if config fails
            "cache_dir": "./model_cache",
            "labels": ["url", "misinfo", "factual"]
        }
    }

logger = logging.getLogger(__name__) # Use standard logger name

# --- Get Configuration ---
CONFIG = get_config()
CLASSIFIER_CONFIG = CONFIG.get('classifier', {}) # Use .get() for safety
# Read parameters from config
MODEL_NAME = CLASSIFIER_CONFIG.get('model_name', 'facebook/bart-large-mnli') # Use config, provide default
CACHE_DIR = CLASSIFIER_CONFIG.get('cache_dir', './model_cache')
LABELS = CLASSIFIER_CONFIG.get('labels', ["url", "misinfo", "factual"]) # Use config labels

# REMOVED unused ID2LABEL / LABEL2ID mappings
# ID2LABEL = {i: label for i, label in enumerate(LABELS)}
# LABEL2ID = {label: i for i, label in enumerate(LABELS)}

# Use consistent naming for the pipeline instance
_classifier_pipeline = None # Renamed from classifier
# REMOVED global tokenizer (not needed for zero-shot pipeline)
# tokenizer = None

def load_classifier() -> bool:
    """
    Loads the zero-shot classification pipeline. Returns True on success.
    Reads configuration from central config.
    """
    global _classifier_pipeline # Use the consistent name
    # Check if already loaded
    if _classifier_pipeline is not None:
        return True
    # Check if transformers library was imported successfully
    if pipeline is None:
         logger.error("Transformers library failed to import. Cannot load classifier.")
         return False

    logger.info(f"Attempting to load classifier model '{MODEL_NAME}' from Hugging Face Hub...")
    try:
        # Ensure cache directory exists if specified
        if CACHE_DIR and not os.path.exists(CACHE_DIR):
             try:
                  os.makedirs(CACHE_DIR)
                  logger.info(f"Created cache directory: {CACHE_DIR}")
             except OSError as e:
                  logger.warning(f"Could not create cache directory {CACHE_DIR}: {e}. Using default Hugging Face cache.")
                  # If dir creation fails, pipeline will use default HF cache

        # Determine device (CPU, CUDA, MPS) - standard logic
        device_type = "cpu"
        device_arg = -1 # Default pipeline device index for CPU
        try:
            if torch.cuda.is_available():
                 device_type = "cuda"
                 device_arg = 0 # Use first CUDA device
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                 # Try loading directly onto MPS device object if possible
                 device_type = "mps"
                 mps_device = torch.device("mps")
                 logger.info("Attempting to load classifier onto MPS device...")
                 _classifier_pipeline = pipeline(
                      "zero-shot-classification",
                      model=MODEL_NAME,
                      tokenizer=MODEL_NAME, # Specify tokenizer explicitly
                      device=mps_device, # Use device object
                      cache_dir=CACHE_DIR
                 )
                 logger.info(f"Zero-shot classifier '{MODEL_NAME}' loaded successfully onto MPS device.")
                 return True # Return early if successful on MPS
        except Exception as mps_err:
            logger.warning(f"Failed to load classifier onto MPS device ({mps_err}). Falling back.")
            device_type = "cpu" # Reset to CPU for standard loading
            device_arg = -1


        # Standard load for CPU/CUDA using device index
        logger.info(f"Using device: {device_type.upper()} (pipeline index: {device_arg}) for classifier model.")
        _classifier_pipeline = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME, # Specify tokenizer explicitly
            device=device_arg, # Use device index
            cache_dir=CACHE_DIR
        )
        logger.info(f"Zero-shot classifier '{MODEL_NAME}' loaded successfully.")
        return True

        # REMOVED commented out sequence classification code block

    except ImportError: # Catch potential errors if transformers wasn't fully available
        logger.error("Transformers library might be incomplete. Failed loading pipeline.", exc_info=True)
    except OSError as e:
        logger.error(f"Model files for '{MODEL_NAME}' not found or download failed. Check model name and network. Cache: '{CACHE_DIR}'. Error: {e}", exc_info=True)
    except Exception as e:
        # Catch-all for other loading errors
        logger.error(f"Failed to load the zero-shot classifier model '{MODEL_NAME}': {e}", exc_info=True)

    # Ensure pipeline is None if loading failed
    _classifier_pipeline = None
    return False

# Use the function name expected by main.py
def classify_intent(query: str) -> Tuple[str, float]:
    """
    Classifies the intent of the query using the loaded zero-shot model.
    Returns the predicted label and a confidence score.
    """
    global _classifier_pipeline # Use consistent name
    if _classifier_pipeline is None:
        logger.error("Classifier pipeline not loaded. Cannot classify intent.")
        # Defaulting to 'misinfo' as defined in main.py's error handling expectation
        logger.warning("Defaulting intent to 'misinfo' due to unavailable classifier.")
        return "misinfo", 0.0 # Return default with zero confidence

    if not query or not isinstance(query, str):
        logger.warning(f"Invalid input for classification: {type(query)}. Returning default 'misinfo'.")
        return "misinfo", 0.0

    try:
        # Truncate input for very long queries - helps prevent errors
        # Bart typically has 1024 token limit, 512 chars is safer heuristic
        max_input_chars = 512
        truncated_query = query[:max_input_chars]
        if len(query) > max_input_chars:
            logger.debug(f"Input query truncated to {max_input_chars} chars for classification.")

        # Ensure LABELS from config are used
        if not LABELS:
             logger.error("Classifier labels not configured. Cannot classify.")
             return "misinfo", 0.0

        results = _classifier_pipeline(truncated_query, candidate_labels=LABELS, multi_label=False)
        predicted_label = results['labels'][0]
        score = results['scores'][0]

        # Ensure label is one of the expected ones (sanity check)
        if predicted_label not in LABELS:
             logger.warning(f"Classifier returned an unexpected label '{predicted_label}'. Mapping to 'other' or default.")
             # Decide how to handle unexpected labels - map to 'other' if exists, or default to misinfo
             return "misinfo", float(score)

        logger.debug(f"Classified intent as: Label='{predicted_label}', Confidence={score:.4f}")
        return predicted_label, float(score) # Return label and score


    except Exception as e:
        logger.error(f"Error during intent classification for query '{query[:50]}...': {e}", exc_info=True)
        # Fallback to 'misinfo' as per original logic
        logger.warning("Returning default intent 'misinfo' due to classification error.")
        return "misinfo", 0.0

# REMOVED classify_query_local function, using classify_intent instead.
