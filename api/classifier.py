# api/classifier.py
import logging
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

logger = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english" # A common lightweight choice
CLASSIFIER_CACHE_DIR = "./model_cache" # Optional: Cache downloaded models

# Define the labels your classifier should predict
LABELS = ["url", "misinfo", "factual"] # Ensure order matches potential model training if you fine-tune
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

classifier = None
tokenizer = None

def load_classifier_model():
    """Loads the classification model and tokenizer only once."""
    global classifier, tokenizer
    if classifier is None:
        try:
            logger.info(f"Loading classification model: {MODEL_NAME}")
            # Ensure cache directory exists if specified
            if CLASSIFIER_CACHE_DIR and not os.path.exists(CLASSIFIER_CACHE_DIR):
                os.makedirs(CLASSIFIER_CACHE_DIR)

            # Load model and tokenizer
            # Note: If the base distilbert model hasn't been fine-tuned for *your specific 3 labels*,
            # its predictions might be less accurate than desired.
            # Using a base sentiment model like sst-2 is a starting point,
            # but fine-tuning on your task (url/misinfo/factual) is ideal for high accuracy.
            # For now, we'll map its outputs conceptually. A better approach would use
            # a model specifically trained or fine-tuned for multi-class intent detection.

            # Simpler approach for now: Use zero-shot classification pipeline
            # This is more flexible if you don't have a fine-tuned model
            classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli", # A decent zero-shot model
                tokenizer="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1, # Use GPU if available
                cache_dir=CLASSIFIER_CACHE_DIR
            )

            logger.info("Classification model loaded successfully.")

            # The code below is for a sequence classification model (if you fine-tune one later)
            # model = AutoModelForSequenceClassification.from_pretrained(
            #     MODEL_NAME,
            #     num_labels=len(LABELS),
            #     id2label=ID2LABEL,
            #     label2id=LABEL2ID,
            #     cache_dir=CLASSIFIER_CACHE_DIR
            # )
            # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CLASSIFIER_CACHE_DIR)
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            # classifier = pipeline(
            #     "text-classification",
            #     model=model,
            #     tokenizer=tokenizer,
            #     device=0 if device == "cuda" else -1 # pipeline expects device index
            # )
            # logger.info(f"Classification pipeline loaded successfully on device: {device}")

        except Exception as e:
            logger.error(f"Error loading classification model: {e}", exc_info=True)
            classifier = None # Ensure it's None if loading fails

def classify_query_local(query: str) -> str:
    """Classifies the query using the local lightweight model."""
    global classifier
    if classifier is None:
        logger.error("Classifier model not loaded. Cannot classify.")
        # Fallback strategy: Could default to 'misinfo' or raise an error
        # Let's default to 'misinfo' for now as it's a primary focus
        return "misinfo"

    try:
        # Using zero-shot pipeline
        results = classifier(query, candidate_labels=LABELS, multi_label=False) # multi_label=False forces single best label
        predicted_label = results['labels'][0]
        # score = results['scores'][0] # You can use the score for confidence if needed
        logger.info(f"Classified query as: {predicted_label}")
        return predicted_label

        # --- Code for sequence classification pipeline (if using a fine-tuned model) ---
        # results = classifier(query, return_all_scores=False) # Get only the top prediction
        # predicted_label = results[0]['label']
        # score = results[0]['score']
        # logger.info(f"Classified query as: {predicted_label} with score: {score:.4f}")
        # return predicted_label
        # --- End sequence classification code ---

    except Exception as e:
        logger.error(f"Error during query classification: {e}", exc_info=True)
        return "misinfo" # Fallback on error
