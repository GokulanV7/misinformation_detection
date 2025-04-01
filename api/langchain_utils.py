# api/langchain_utils.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings # Still correct
# Ensure community embeddings are available if needed, but HF is likely community now
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# RetrievalQA is older, consider newer LCEL chains later if optimizing further
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, DirectoryLoader, WebBaseLoader # Keep if needed elsewhere
import logging
import os
from typing import List, Dict, Any # Use Any for flexible dicts
import yaml
# Correct relative import for query_groq (we need it for the RAG query part)
from .groq_utils import query_groq
import re

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

def preprocess_text(text: str) -> str:
    """Basic text preprocessing."""
    if not isinstance(text, str): # Add type check
        logger.warning(f"preprocess_text received non-string input: {type(text)}")
        return ""
    text = text.lower()
    # Keep basic punctuation that might be relevant for meaning? Adjust as needed.
    text = re.sub(r'[^a-z0-9\s.,?!:\'-]', '', text) # Allow some punctuation
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    return text

class RealTimeDataProcessor:

    def __init__(self, api_key: str, embedding_model_name: str = None,
                  index_path: str = None):
        self.api_key = api_key # Groq key needed for query_rag

        # Load defaults from config if not provided
        rag_config = config.get('rag', {})
        self.embedding_model_name = embedding_model_name or rag_config.get('embedding_model', "sentence-transformers/all-mpnet-base-v2")
        self.index_path = index_path or os.getenv("FAISS_INDEX_PATH", "data/rag_data/default_index") # Use env var fallback

        self.embeddings = None
        self.db = None
        self._load_embeddings()
        self._load_vector_db()


    def _load_embeddings(self):
        """Loads the embedding model."""
        try:
            # Check if community package path is needed based on langchain version
            # For newer Langchain, HuggingFaceEmbeddings might be in community
            try:
                 from langchain_community.embeddings import HuggingFaceEmbeddings
                 logger.info("Using HuggingFaceEmbeddings from langchain_community.")
            except ImportError:
                 from langchain.embeddings import HuggingFaceEmbeddings
                 logger.info("Using HuggingFaceEmbeddings from core langchain.")

            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
            logger.info(f"Embedding model '{self.embedding_model_name}' loaded.")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{self.embedding_model_name}': {e}", exc_info=True)
            self.embeddings = None # Ensure it's None on failure

    def _load_vector_db(self):
        """Loads the FAISS index if it exists."""
        if self.embeddings is None:
             logger.error("Cannot load vector DB: Embeddings failed to load.")
             return

        if os.path.exists(self.index_path):
            try:
                # Allow dangerous deserialization, but log a warning
                logger.warning(f"Loading FAISS index from {self.index_path} with allow_dangerous_deserialization=True. Ensure the index file source is trusted.")
                self.db = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True # Required for pickle-based FAISS indexes
                )
                logger.info(f"Loaded existing FAISS index from {self.index_path}")
            except Exception as e:
                 logger.error(f"Error loading FAISS index from {self.index_path}: {e}", exc_info=True)
                 self.db = None
        else:
            self.db = None
            logger.info(f"No existing FAISS index found at {self.index_path}. Index will be created on first update.")

    def update_index(self, new_data: List[str]):
        """
        Update the FAISS index with new data. Creates index if it doesn't exist.
        """
        if self.embeddings is None:
             logger.error("Cannot update index: Embeddings not loaded.")
             return # Or raise error

        if not new_data:
            logger.info("No new data provided to update_index.")
            return

        try:
            # Preprocess data
            processed_data = [preprocess_text(text) for text in new_data if text] # Filter out empty strings
            if not processed_data:
                 logger.info("No valid text data left after preprocessing.")
                 return

            # Split documents
            rag_config = config.get('rag', {})
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=rag_config.get('chunk_size', 1000),
                chunk_overlap=rag_config.get('chunk_overlap', 200)
            )
            chunks = text_splitter.create_documents(processed_data)

            if not chunks:
                logger.warning("Text splitting resulted in no chunks.")
                return

            # Create or update index
            if self.db is None:
                logger.info(f"Creating new FAISS index at {self.index_path}")
                self.db = FAISS.from_documents(chunks, self.embeddings)
            else:
                logger.info(f"Adding {len(chunks)} new chunks to existing FAISS index.")
                self.db.add_documents(chunks)

            # Ensure directory exists before saving
            index_dir = os.path.dirname(self.index_path)
            if index_dir and not os.path.exists(index_dir):
                 os.makedirs(index_dir)
                 logger.info(f"Created directory for FAISS index: {index_dir}")

            self.db.save_local(self.index_path)
            logger.info(f"FAISS index updated and saved to {self.index_path}")

        except Exception as e:
            logger.error(f"Error updating FAISS index: {e}", exc_info=True)
            # Decide if you want to raise the error or just log it
            # raise

    async def retrieve_context(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieves relevant context from the vector store."""
        if self.db is None:
            logger.warning("No FAISS index available for context retrieval.")
            return []
        if self.embeddings is None:
             logger.error("Cannot retrieve context: Embeddings not loaded.")
             return []

        try:
            # Maybe preprocess query slightly?
            # processed_query = preprocess_text(query)
            docs = self.db.similarity_search(query, k=top_k)
            context = [doc.page_content for doc in docs]
            logger.info(f"Retrieved {len(context)} context snippets for query: {query[:50]}...")
            # Optionally log the context itself if needed for debugging (can be verbose)
            # logger.debug(f"Retrieved context: {context}")
            return context
        except Exception as e:
            logger.error(f"Error during context retrieval: {e}", exc_info=True)
            return []

    async def query_rag(self, query: str, model: str, use_for: str = "general") -> Dict[str, Any]:
        """
        Queries the RAG system. Uses context + Groq.
        'use_for' hint can tailor the prompt slightly.
        """
        if self.db is None:
             logger.warning("RAG query attempted but vector DB is not loaded.")
             return {"answer": "I cannot answer using real-time data as the knowledge base isn't available.", "source": "RAG Error - No DB"}

        try:
            context = await self.retrieve_context(query)
            if not context:
                logger.info(f"No relevant context found in RAG for query: {query[:50]}...")
                # For misinfo, maybe *don't* proceed without context?
                # For factual fallback, definitely don't proceed.
                if use_for == "misinfo_check":
                    # If checking misinfo and NO context found, maybe Groq alone isn't useful?
                    # Return a specific message indicating lack of grounding.
                    return {
                        "category": "uncertain", "confidence": 0.3,
                        "explanation": "Could not find relevant real-time information to verify this statement.",
                        "recommendation": "Verify with trusted sources independently.",
                        "source": "RAG - No Context Found"
                    }
                else: # Factual fallback or general
                    return {"answer": "I couldn't find specific real-time information related to your question.", "source": "RAG - No Context Found"}


            context_str = "\n---\n".join(context) # Separator for clarity

            # Tailor prompt based on intended use
            if use_for == "misinfo_check":
                 prompt = f"""Please analyze the following statement based *primarily* on the provided real-time context. Assess if the statement aligns with, contradicts, or is not mentioned in the context.

Context from recent news:
{context_str}
---
Statement to Analyze: "{query}"

Provide your analysis in this JSON format:
{{
    "category": "[truth/fake/uncertain/opinion/etc.]",
    "confidence": [0.0-1.0],
    "explanation": "[Explain how context supports/refutes the statement, or why it's uncertain]",
    "context_match": "[yes/no/partial]",
    "key_context_points": ["Relevant snippet 1", "..."]
}}"""
                 temp = 0.1 # Be factual for analysis

            elif use_for == "factual_fallback":
                 prompt = f"""Answer the following question using the provided recent context. If the context doesn't directly answer it, state that.

Context:
{context_str}
---
Question: "{query}"

Answer directly:"""
                 temp = 0.2 # Slightly more flexible for answering
            else: # General RAG query (should be less common now)
                 prompt = f"""Use the following context to answer the question at the end. If you can't find the answer in the context, just say that you don't know based on the provided information. Do not use prior knowledge.
Context:
{context_str}
---
Question: "{query}"
Answer:"""
                 temp = 0.1

            # Query Groq using the context
            groq_response = await query_groq(prompt, self.api_key, model, temperature=temp)

            answer = groq_response.get("choices", [{}])[0].get("message", {}).get("content", "")

            # If used for misinfo check, try to parse the expected JSON
            if use_for == "misinfo_check":
                try:
                    rag_analysis = json.loads(answer)
                    rag_analysis["source"] = "RAG (News Context + Groq Analysis)"
                    return rag_analysis
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON response from RAG misinfo check: {answer[:200]}...")
                    # Fallback to a basic structure if JSON fails
                    return {
                        "category": "uncertain", "confidence": 0.4,
                        "explanation": "AI analysis based on context was unclear: " + answer[:300],
                        "recommendation": "Manual verification recommended.",
                        "source": "RAG (News Context + Groq Analysis - Format Error)"
                    }
            else: # Factual fallback or general
                return {"answer": answer, "source": "RAG (News Context + Groq Answer)", "retrieved_context": context[:1]} # Maybe include snippet of context used

        except Exception as e:
            logger.error(f"RAG query failed: {e}", exc_info=True)
            return {"answer": f"An error occurred while consulting real-time data: {e}", "source": "RAG Error"}


# Keep load_data_from_web if used elsewhere, but ensure it uses httpx if fetching web pages directly
# Example of updating load_data_from_web to use httpx (if needed)
async def load_text_from_web_url(url: str, timeout: int = 15) -> str:
    """Fetches text content from a single URL asynchronously."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            # Basic text extraction, consider BeautifulSoup for better HTML parsing
            # This assumes content-type is text-based
            return response.text
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Error {e.response.status_code} fetching {url}: {e.response.text[:200]}")
    except httpx.RequestError as e:
        logger.error(f"Request Error fetching {url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {e}", exc_info=True)
    return ""
