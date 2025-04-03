import logging
import pickle
import os
from typing import List, Dict, Any, Set, Optional, Tuple

import networkx as nx
import spacy
from spacy.language import Language

from .utils import get_config

logger = logging.getLogger(__name__)
CONFIG = get_config()
KG_CONFIG = CONFIG.get('knowledge_graph', {})
KG_STORAGE_PATH = KG_CONFIG.get('storage_path', 'data/kg_store/knowledge_graph.gpickle')
ENTITY_TYPES = set(KG_CONFIG.get('entity_types', ["PERSON", "ORG", "GPE"])) # Default focus
QUERY_DEPTH = KG_CONFIG.get('query_depth', 1)

# Global variables for graph and NLP model (loaded once)
graph: Optional[nx.DiGraph] = None
nlp: Optional[Language] = None

def load_spacy_model(model_name="en_core_web_sm") -> Optional[Language]:
    """Loads the spaCy model for NER."""
    global nlp
    if nlp:
        return nlp
    try:
        # Consider disabling components not needed for NER for speed
        nlp = spacy.load(model_name, disable=["parser", "lemmatizer", "attribute_ruler", "tagger"])
        logger.info(f"SpaCy NER model '{model_name}' loaded successfully.")
        return nlp
    except ImportError:
        logger.error("SpaCy library not installed. Cannot perform NER.")
    except OSError:
        logger.error(f"SpaCy model '{model_name}' not found. Download it: python -m spacy download {model_name}")
    except Exception as e:
        logger.error(f"Error loading spaCy model: {e}", exc_info=True)
    return None

def load_graph() -> nx.DiGraph:
    """Loads the knowledge graph from storage or creates a new one."""
    global graph
    if graph:
        return graph

    graph_dir = os.path.dirname(KG_STORAGE_PATH)
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
        logger.info(f"Created directory for Knowledge Graph: {graph_dir}")

    if os.path.exists(KG_STORAGE_PATH):
        try:
            with open(KG_STORAGE_PATH, 'rb') as f:
                graph = pickle.load(f)
            logger.info(f"Knowledge Graph loaded from {KG_STORAGE_PATH} with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        except Exception as e:
            logger.error(f"Error loading knowledge graph from {KG_STORAGE_PATH}: {e}. Creating a new graph.")
            graph = nx.DiGraph()
    else:
        logger.info(f"No existing knowledge graph found at {KG_STORAGE_PATH}. Creating a new graph.")
        graph = nx.DiGraph()

    if not isinstance(graph, nx.DiGraph): # Handle case where loading failed somehow
        logger.warning("Loaded object is not a DiGraph, creating a new one.")
        graph = nx.DiGraph()

    return graph

def save_graph():
    """Saves the current knowledge graph to storage."""
    global graph
    if graph is None:
        logger.warning("Graph not loaded, cannot save.")
        return
    try:
        with open(KG_STORAGE_PATH, 'wb') as f:
            pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
        logger.debug(f"Knowledge Graph saved to {KG_STORAGE_PATH}")
    except Exception as e:
        logger.error(f"Error saving knowledge graph to {KG_STORAGE_PATH}: {e}", exc_info=True)

def extract_entities(text: str) -> List[Tuple[str, str]]:
    """Extracts named entities relevant to the configured types from text."""
    global nlp
    if not text or not nlp:
        return []

    try:
        doc = nlp(text[:nlp.max_length]) # Process text respecting model limits
        entities = []
        for ent in doc.ents:
            if ent.label_ in ENTITY_TYPES:
                entities.append((ent.text.strip(), ent.label_))
        # Simple deduplication by text
        return list(dict.fromkeys(entities))
    except Exception as e:
        logger.error(f"Error during entity extraction: {e}", exc_info=True)
        return []


def add_claim_to_graph(claim_text: str, assessment: str, source_url: str, entities: List[Tuple[str, str]]):
    """Adds a claim and its entities to the knowledge graph."""
    global graph
    if not graph:
        load_graph() # Ensure graph is loaded
        if not graph :
             logger.error("Cannot add claim, KG failed to load."); return

    # Normalize assessment for graph property
    assessment_norm = assessment.lower().replace(" ", "_")

    # Create a unique node for the claim itself
    claim_id = f"claim:{hash(claim_text)}_{source_url[:50]}" # Simple unique ID
    graph.add_node(claim_id, type='claim', text=claim_text[:200], assessment=assessment_norm, url=source_url)

    entity_nodes = set()
    for entity_text, entity_type in entities:
        entity_id = f"{entity_type.lower()}:{entity_text.lower().replace(' ', '_')}" # Normalize node ID
        if not graph.has_node(entity_id):
            graph.add_node(entity_id, type=entity_type, name=entity_text)
            logger.debug(f"Added new entity node to KG: {entity_id} ({entity_text})")
        else: # Increment mention count or update properties if needed
            mention_count = graph.nodes[entity_id].get('mention_count', 0)
            graph.nodes[entity_id]['mention_count'] = mention_count + 1

        # Link the claim to the entity (claim mentions entity)
        graph.add_edge(claim_id, entity_id, relationship='mentions')
        # Link the entity to the claim (entity mentioned_in claim) - easier querying?
        graph.add_edge(entity_id, claim_id, relationship='mentioned_in')
        entity_nodes.add(entity_id)

    logger.info(f"Updated KG: Added claim '{claim_id}' linked to {len(entity_nodes)} entities.")
    save_graph()


def query_kg_for_entities(entities: List[Tuple[str, str]]) -> str:
    """Queries the KG for context about given entities."""
    global graph
    if not graph or not entities:
        return "No relevant entity information found in Knowledge Graph."

    insights = []
    for entity_text, entity_type in entities:
        entity_id = f"{entity_type.lower()}:{entity_text.lower().replace(' ', '_')}"
        if graph.has_node(entity_id):
            entity_insights = []
            # Find claims linked TO this entity
            neighbors = list(nx.neighbors(graph, entity_id)) # Nodes this entity points to
            claims_mentioned_in = [n for n in neighbors if graph.nodes[n].get('type') == 'claim']

            if claims_mentioned_in:
                assessments = [graph.nodes[claim_id].get('assessment', 'unknown') for claim_id in claims_mentioned_in]
                assessment_counts = {ass: assessments.count(ass) for ass in set(assessments)}
                insight_str = f"Entity '{entity_text}' is mentioned in {len(claims_mentioned_in)} previous claims with assessments: {assessment_counts}."
                entity_insights.append(insight_str)

            # Add more complex queries if needed (e.g., relations between entities)

            if entity_insights:
                insights.extend(entity_insights)

    if not insights:
        return "Entities found in text, but no prior claims recorded in Knowledge Graph."

    # Limit the length of the insights string
    final_insight_str = "Knowledge Graph Context: " + " ".join(insights)
    return final_insight_str[:500] # Limit output length
