import logging
import os
from typing import List, Tuple, Optional, Dict, Any
import pickle

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# If using langchain-cohere directly:
from langchain_cohere import CohereRerank # Check correct import
# If calling Cohere API directly via SDK:
import cohere

from .utils import get_config
from .groq_utils import query_groq

# Load config globally
CONFIG = get_config()
RAG_CONFIG = CONFIG['rag']
COHERE_CONFIG = CONFIG.get('cohere', {})
GROQ_CONFIG = CONFIG['groq']

# Initialize Cohere client if calling API directly
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = None
if COHERE_API_KEY:
    try:
        co = cohere.Client(COHERE_API_KEY)
    except Exception as e:
        logging.error(f"Failed to initialize Cohere client: {e}")

logger = logging.getLogger(__name__)

class RealTimeDataProcessor:
    """Handles RAG indexing, retrieval, and augmented querying."""

    def __init__(self):
        self.config = CONFIG # Use global config
        self.rag_config = RAG_CONFIG
        self.index_path = self.rag_config['index_path']
        self.embedding_model_name = self.rag_config['embedding_model']
        self.chunk_size = self.rag_config['chunk_size']
        self.chunk_overlap = self.rag_config['chunk_overlap']
        self.retrieval_multiplier = self.rag_config.get('retrieval_multiplier', 3) # Default multiplier
        self.final_top_k = COHERE_CONFIG.get('rerank_top_n', 3) # Use Cohere config for final count

        self._ensure_dir_exists(self.index_path)
        self.embeddings = self._load_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        self.vector_store = self._load_or_initialize_vector_store()
        # Langchain CohereRerank integration (optional, check docs)
        # self.reranker = self._load_reranker() # If using LC integration


    def _ensure_dir_exists(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"Created directory: {path}")

    def _load_embeddings(self) -> Optional[HuggingFaceEmbeddings]:
        """Loads the sentence transformer embedding model."""
        try:
            # Use cache_folder to align with classifier caching if desired
            # model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'} # Basic device selection
            # Use device based on torch availability (cpu/cuda/mps) - Requires torch installed
            device = "cpu"
            try:
                import torch
                if torch.cuda.is_available(): device = "cuda"
                elif torch.backends.mps.is_available(): device = "mps" # For Apple Silicon
            except ImportError: pass # Stick to CPU if torch not installed

            encode_kwargs = {'normalize_embeddings': True} # Important for cosine similarity
            logger.info(f"Loading embedding model: {self.embedding_model_name} onto device: {device}")

            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': device},
                encode_kwargs=encode_kwargs,
                cache_folder=CONFIG['classifier']['cache_dir'] # Use same cache dir
            )
            logger.info("Embedding model loaded.")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            return None

    def _load_or_initialize_vector_store(self) -> Optional[FAISS]:
        """Loads FAISS index from disk or initializes a new one."""
        index_file = os.path.join(self.index_path, "index.faiss")
        pkl_file = os.path.join(self.index_path, "index.pkl")

        if os.path.exists(index_file) and os.path.exists(pkl_file) and self.embeddings:
            try:
                # FAISS.load_local requires allow_dangerous_deserialization=True
                # Ensure you trust the source of index.pkl
                logger.info(f"Loading existing FAISS index from {self.index_path}...")
                vector_store = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("FAISS index loaded successfully.")
                return vector_store
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}. Re-initializing.", exc_info=True)
                # Fall through to initialize a new one if loading fails

        # Initialize new if files don't exist or loading failed
        if self.embeddings:
            logger.info("Initializing a new FAISS index.")
            # Create a dummy index to allow saving empty initially
            dummy_doc = Document(page_content="init")
            try:
                 vs = FAISS.from_documents([dummy_doc], self.embeddings)
                 # Immediately delete the dummy doc - bit hacky but works for Langchain FAISS
                 ids_to_delete = list(vs.index_to_docstore_id.values())
                 vs.delete(ids=ids_to_delete)
                 logger.info("New FAISS index initialized.")
                 vs.save_local(self.index_path) # Save the empty structure
                 return vs
            except Exception as e:
                 logger.error(f"Failed to initialize FAISS index: {e}", exc_info=True)
                 return None

        logger.error("Cannot initialize vector store without embedding model.")
        return None

    def update_index(self, documents: List[Document]):
        """Adds new documents to the FAISS index and saves it."""
        if not self.vector_store or not self.embeddings:
            logger.error("Vector store or embeddings not initialized. Cannot update index.")
            return False

        if not documents:
            logger.warning("No documents provided to update index.")
            return True # No error, just nothing to do

        logger.info(f"Splitting {len(documents)} documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Generated {len(chunks)} chunks for indexing.")

        if not chunks:
            logger.warning("No chunks generated after splitting documents.")
            return True

        # Check existing IDs to potentially avoid duplicates if metadata allows
        # existing_ids = list(self.vector_store.index_to_docstore_id.values()) # If needed

        try:
            logger.info(f"Adding {len(chunks)} new chunks to the FAISS index...")
            # Filter out chunks with potentially empty content after splitting
            valid_chunks = [chunk for chunk in chunks if chunk.page_content and chunk.page_content.strip()]
            if not valid_chunks:
                 logger.warning("All chunks were empty after splitting/validation.")
                 return True

            chunk_ids = self.vector_store.add_documents(valid_chunks)
            logger.info(f"Successfully added {len(chunk_ids)} new chunks to index.")

            # --- Persist changes ---
            self.vector_store.save_local(self.index_path)
            logger.info(f"FAISS index saved successfully to {self.index_path}")
            return True

        except Exception as e:
            logger.error(f"Error updating FAISS index: {e}", exc_info=True)
            return False


    def retrieve_context(self, query: str) -> List[Document]:
         """Retrieves relevant document chunks based on the query."""
         if not self.vector_store:
              logger.error("Vector store not available for retrieval.")
              return []

         try:
             k = self.final_top_k * self.retrieval_multiplier
             logger.debug(f"Performing similarity search for query '{query}' with k={k}")
             # Use similarity search; other methods like MMR exist
             results = self.vector_store.similarity_search(query, k=k)
             logger.debug(f"Retrieved {len(results)} initial documents.")
             return results
         except Exception as e:
              logger.error(f"Error during vector store retrieval: {e}", exc_info=True)
              return []

    async def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
         """Re-ranks documents using Cohere API."""
         if not co or not COHERE_API_KEY or not documents:
              logger.warning("Cohere client not available or no documents to rerank. Returning original order.")
              return documents[:self.final_top_k] # Return top N originals if no rerank

         if not COHERE_CONFIG.get('rerank_model'):
             logger.warning("Cohere rerank model not configured. Returning original order.")
             return documents[:self.final_top_k]

         logger.debug(f"Reranking {len(documents)} documents for query: '{query}' using Cohere '{COHERE_CONFIG['rerank_model']}'")
         doc_texts = [doc.page_content for doc in documents]

         try:
             # Using Cohere SDK directly
             rerank_response = co.rerank(
                  model=COHERE_CONFIG['rerank_model'],
                  query=query,
                  documents=doc_texts,
                  top_n=self.final_top_k # Ask Cohere to return only the best N
             )
             # Map reranked results back to original Document objects
             reranked_docs = []
             # result format {index: int, relevance_score: float}
             for result in rerank_response.results:
                 if result.relevance_score > 0.1: # Optional threshold
                      reranked_docs.append(documents[result.index])
                 else:
                     logger.debug(f"Dropping reranked doc index {result.index} due to low score ({result.relevance_score:.3f})")

             # Using LangChain integration (if preferred and setup)
             # reranker_instance = CohereRerank(cohere_api_key=COHERE_API_KEY, model=COHERE_CONFIG['rerank_model'], top_n=self.final_top_k)
             # reranked_docs = reranker_instance.compress_documents(documents=documents, query=query)

             logger.info(f"Cohere reranked {len(documents)} -> {len(reranked_docs)} documents.")
             return reranked_docs

         except Exception as e:
             logger.error(f"Error during Cohere reranking: {e}", exc_info=True)
             # Fallback to original top N documents if reranking fails
             return documents[:self.final_top_k]


    async def query_rag(self, user_query: str, use_for: str = "misinfo_check") -> Tuple[Optional[str], Optional[List[Dict]]]:
        """
        Performs RAG: Retrieves, Re-ranks, checks sufficiency, and Queries LLM.

        Args:
            user_query: The user's question or statement.
            use_for: Hint for prompt construction ('misinfo_check' or 'factual_qa').

        Returns:
            A tuple: (LLM response text or None, List of source document dicts or None)
        """
        if not self.vector_store:
            logger.error("RAG query failed: Vector store not initialized.")
            return None, None

        # 1. Retrieve initial context
        initial_documents = self.retrieve_context(user_query)
        if not initial_documents:
            logger.warning(f"RAG: No initial documents found for query: {user_query}")
            return None, None # Signal no context found

        # 2. Re-rank documents
        reranked_documents = await self.rerank_documents(user_query, initial_documents)
        if not reranked_documents:
            logger.warning(f"RAG: No documents remaining after re-ranking for query: {user_query}")
            return None, None

        context_str = "\n\n---\n\n".join([doc.page_content for doc in reranked_documents])
        sources = [{"source": doc.metadata.get('source', 'Unknown'),
                    "snippet": doc.page_content[:150] + "..."} # Short snippet for context
                   for doc in reranked_documents]


        # 3. Check if RAG context is sufficient (Using Groq for evaluation)
        sufficiency_prompt = GROQ_CONFIG['check_rag_sufficiency_prompt'].format(
            query=user_query, context=context_str
        )
        sufficiency_check_start_time = asyncio.get_event_loop().time()
        try:
            # Use a fast, small model for this check if possible/configured
            sufficiency_response = await query_groq(sufficiency_prompt, temperature=0.0, model="llama3-8b-8192") # Example fast model
            logger.debug(f"RAG Sufficiency check took {asyncio.get_event_loop().time() - sufficiency_check_start_time:.2f}s")

            if sufficiency_response:
                 sufficiency_answer = sufficiency_response.strip().upper().splitlines()[0]
                 logger.info(f"RAG Context Sufficiency Assessment: {sufficiency_answer}")
                 # Check if starts with NO or PARTIALLY
                 if sufficiency_answer.startswith("NO") or sufficiency_answer.startswith("PARTIALLY"):
                       logger.warning(f"RAG context deemed insufficient by LLM for query: '{user_query}'. Falling back.")
                       return None, None # Indicate insufficient context
            else:
                 logger.warning("RAG sufficiency check LLM call failed. Assuming context might be sufficient.")
                 # Proceed cautiously if check fails

        except Exception as e:
             logger.error(f"Error during RAG sufficiency check LLM call: {e}", exc_info=True)
             # Proceed cautiously, assume might be sufficient if check errors out
             logger.warning("Proceeding with RAG despite sufficiency check error.")


        # 4. Query LLM with augmented prompt (if context deemed sufficient)
        # Build prompt based on use_for
        # Simplified prompt building (can be expanded based on groq_utils logic)
        if use_for == "misinfo_check":
             # Need a prompt asking to analyze the query based *only* on the context
             prompt = f"""Analyze the following statement based *only* on the provided context documents.
Determine if the statement is supported, contradicted, or if the context doesn't provide enough information. Explain your reasoning.
Statement: "{user_query}"
Context Documents:
{context_str}
Analysis:""" # TODO: Refine this prompt for misinfo check

        elif use_for == "factual_qa":
             prompt = f"""Answer the following question based *only* on the provided context documents.
If the context doesn't contain the answer, state that clearly.
Question: "{user_query}"
Context Documents:
{context_str}
Answer:"""
        else:
             logger.warning(f"Unknown 'use_for' value: {use_for}. Using generic prompt.")
             prompt = f"""Based on the following context, respond to the query: "{user_query}" \nContext:\n{context_str}\nResponse:"""

        logger.debug(f"Querying Groq with RAG context for: {user_query}")
        llm_response = await query_groq(prompt, temperature=self.config['groq']['temperature'], model=self.config['groq']['model'])

        return llm_response, sources

    # Helper function to load data (e.g., from MongoDB if scraper saves there)
    def load_data_from_mongo(self) -> List[Document]:
         """Loads data from MongoDB (adapt based on org12.py structure)."""
         docs = []
         mongo_uri = os.getenv("MONGO_URI")
         db_name = CONFIG.get("mongo", {}).get("db_name")
         collection_name = CONFIG.get("mongo", {}).get("collection_name")

         if not all([mongo_uri, db_name, collection_name]):
              logger.warning("MongoDB config missing, cannot load data for RAG update.")
              return []

         try:
              from pymongo import MongoClient
              client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
              db = client[db_name]
              collection = db[collection_name]

              # Load recent articles, perhaps based on 'scrape_timestamp'
              # Define your query here, e.g., only load articles not yet processed for RAG
              mongo_results = collection.find({}, {"_id": 0, "title": 1, "content": 1, "url": 1, "source_domain": 1, "date_published": 1}).limit(100) # Example: Limit loading

              for item in mongo_results:
                  if item.get('content'):
                       # Construct LangChain Document
                       metadata = {
                           "source": item.get('url', item.get('source_domain', 'Unknown')),
                           "title": item.get('title', 'No Title'),
                           "publish_date": str(item.get('date_published', '')),
                           # Add other relevant metadata
                       }
                       doc = Document(page_content=item['content'], metadata=metadata)
                       docs.append(doc)
              client.close()
              logger.info(f"Loaded {len(docs)} documents from MongoDB collection '{collection_name}'.")

         except ImportError:
             logger.error("Pymongo not installed. Cannot load data from MongoDB.")
         except Exception as e:
              logger.error(f"Error loading data from MongoDB: {e}", exc_info=True)

         return docs

    def run_periodic_update(self):
        """Loads data (e.g., from Mongo) and updates the index."""
        logger.info("Starting periodic RAG index update...")
        new_documents = self.load_data_from_mongo() # Or load from another source
        if new_documents:
            success = self.update_index(new_documents)
            if success:
                logger.info("Periodic RAG index update completed successfully.")
            else:
                logger.error("Periodic RAG index update failed.")
        else:
            logger.info("No new documents found for periodic RAG update.")
