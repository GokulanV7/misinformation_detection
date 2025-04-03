import argparse
import logging
import os
import sys
from typing import List

import pandas as pd
from langchain_core.documents import Document

# Ensure API directory is in path if running script from project root
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
# Instead: Make Langchain utils runnable independently or import carefully
try:
    from api.langchain_utils import RealTimeDataProcessor
    from api.utils import get_config, setup_logging
except ImportError:
    print("Error: Could not import necessary modules from the 'api' directory.")
    print("Ensure you run this script from the project root directory or that the 'api' package is correctly installed/discoverable.")
    sys.exit(1)

setup_logging() # Use logging config from main app
logger = logging.getLogger(__name__)

def create_documents_from_csv(csv_path: str, text_column: str, metadata_columns: List[str]) -> List[Document]:
    """Reads a CSV and converts rows into LangChain Documents."""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Read {len(df)} rows from {csv_path}")
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_path}: {e}", exc_info=True)
        return []

    documents = []
    required_columns = [text_column] + metadata_columns
    if not all(col in df.columns for col in required_columns):
        logger.error(f"CSV missing required columns. Needed: {required_columns}, Found: {list(df.columns)}")
        return []

    # Fill NaN values in metadata columns to avoid errors
    df[metadata_columns] = df[metadata_columns].fillna('N/A')
    df[text_column] = df[text_column].fillna('') # Fill NaN text as empty string


    for index, row in df.iterrows():
        page_content = row[text_column]
        if not isinstance(page_content, str) or not page_content.strip():
            logger.warning(f"Skipping row {index} due to empty or invalid text content.")
            continue

        metadata = {col: str(row[col]) for col in metadata_columns} # Ensure metadata vals are strings
        # Standardize the 'source' metadata if possible (e.g., from a 'url' column)
        if 'url' in metadata:
            metadata['source'] = metadata['url']
        elif 'source_domain' in metadata:
            metadata['source'] = metadata['source_domain']
        else:
            # fallback or define explicitly
            metadata.setdefault('source', os.path.basename(csv_path)) # Use filename as fallback source


        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    logger.info(f"Created {len(documents)} LangChain documents from CSV.")
    return documents


def main():
    parser = argparse.ArgumentParser(description="Ingest data from a CSV file into the RAG FAISS index.")
    parser.add_argument("csv_path", help="Path to the input CSV file.")
    parser.add_argument("--text_col", default="content", help="Name of the column containing the main text content.")
    parser.add_argument("--meta_cols", nargs='+', default=["url", "title", "date_published", "source_domain"],
                        help="List of column names to include as metadata (space-separated). 'url' or 'source_domain' recommended.")
    # Allow overriding index path for flexibility if needed
    parser.add_argument("--index_path", default=None, help="Override the index path from config.yaml.")

    args = parser.parse_args()

    logger.info("--- Starting CSV to RAG Ingestion ---")
    config = get_config() # Load config to ensure paths/models match API

    # Initialize RAG processor
    rag_processor = RealTimeDataProcessor()
    if args.index_path: # Override path if provided via args
        logger.info(f"Overriding index path to: {args.index_path}")
        rag_processor.index_path = args.index_path
        # Force reload/reinit vector store with the new path if needed
        rag_processor.vector_store = rag_processor._load_or_initialize_vector_store()


    if not rag_processor.embeddings or not rag_processor.vector_store:
        logger.error("Failed to initialize RAG components. Aborting.")
        sys.exit(1)

    # Process CSV
    documents = create_documents_from_csv(args.csv_path, args.text_col, args.meta_cols)

    if not documents:
        logger.warning("No documents were created from the CSV. Ingestion finished.")
        sys.exit(0)

    # Update index
    success = rag_processor.update_index(documents)

    if success:
        logger.info("--- CSV to RAG Ingestion Completed Successfully ---")
    else:
        logger.error("--- CSV to RAG Ingestion Failed ---")
        sys.exit(1)


if __name__ == "__main__":
    main()
