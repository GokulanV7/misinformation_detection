# api/models.py
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Literal, Any, Union

# --- Request Models ---
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The text (claim, question, or message with URL) to analyze.")

# --- Sub-Models for Responses ---
class EvidenceItem(BaseModel):
    source: str = Field(..., description="URL, Document Title, or source type (e.g., LLM internal)")
    snippet: Optional[str] = Field(None, description="Relevant text snippet from the source.")
    assessment_note: Optional[str] = Field(None, description="Note on relevance (e.g., 'Supports', 'Contradicts', 'Retrieved Context')")

class TextContextAssessment(BaseModel):
    suspicion_level: Literal["High", "Medium", "Low", "N/A"]
    key_indicators: List[str] = Field(default_factory=list, description="Keywords or patterns detected.")

class ScanResultDetail(BaseModel):
    status: str = Field(..., description="Outcome of the scan ('success', 'error', 'no_data', 'pending', 'rate_limited', 'skipped', 'no_scan_found')")
    details: Optional[Dict[str, Any]] = Field(None, description="Parsed findings from the scanner, structure varies by scanner.")

class UrlScanResults(BaseModel):
    virustotal: Optional[ScanResultDetail] = None
    ipqualityscore: Optional[ScanResultDetail] = None
    urlscanio: Optional[ScanResultDetail] = None

# --- Main Response Models ---
class BaseAnalysisResponse(BaseModel):
    request_id: str = Field(..., description="Unique identifier for the analysis request.")
    input_text: str = Field(..., description="The original input text provided by the user.")
    processing_time_ms: float = Field(..., description="Total time taken for the analysis in milliseconds.")
    assessment: str = Field(..., description="The overall assessment category determined by the analysis.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="System's confidence in the assessment (0.0 to 1.0).")

    # Ensure confidence score is always clamped
    @validator('confidence_score')
    def clamp_confidence(cls, v):
        return min(max(v, 0.0), 1.0)

# Forward references are implicitly handled by Pydantic v2+
class UrlAnalysisResponse(BaseAnalysisResponse):
    assessment: Literal["Malicious", "Phishing", "Spam", "Suspicious", "Likely Safe", "Uncertain", "Analysis Failed"]
    scanned_url: Optional[str] = Field(None, description="The primary URL that was extracted and analyzed.")
    analysis_summary: str = Field(..., description="A brief textual summary of the findings.")
    text_context_assessment: Optional[TextContextAssessment] = Field(None, description="Assessment of text surrounding the URL, if applicable.")
    scan_results: Optional[UrlScanResults] = Field(None, description="Detailed results from integrated URL scanning services.")
    evidence_notes: List[str] = Field(default_factory=list, description="Specific textual points supporting the assessment.")

class FactualAnalysisResponse(BaseAnalysisResponse):
    assessment: Literal["Likely Factual", "Opinion", "Needs Verification / Uncertain", "Contradictory Information Found"]
    answer: Optional[str] = Field(None, description="The direct answer or summary provided.")
    explanation: Optional[str] = Field(None, description="Supporting explanation or reasoning for the answer/assessment.")
    data_source: Literal["RAG", "LLM Internal Knowledge", "Web Search", "N/A"] # Added N/A
    supporting_evidence: List[EvidenceItem] = Field(default_factory=list, description="List of sources or context snippets used.")
    knowledge_graph_insights: Optional[str] = Field(None, description="Contextual insights from the knowledge graph regarding entities.")
    related_url_analysis: Optional[UrlAnalysisResponse] = Field(None, description="Analysis results if a relevant URL was present in the input.")

class MisinformationAnalysisResponse(BaseAnalysisResponse):
    assessment: Literal["Likely Factual", "Likely Misleading", "Opinion", "Needs Verification / Uncertain", "Contradictory Information Found"]
    explanation: str = Field(..., description="Explanation justifying the assessment, potentially including reasoning and identified issues.")
    evidence: List[EvidenceItem] = Field(default_factory=list, description="Supporting evidence (quotes, sources, context snippets).")
    key_issues_identified: List[str] = Field(default_factory=list, description="Specific potential problems identified by the LLM (e.g., 'Unsourced Claim').") # Use default_factory
    verifiable_claims: List[str] = Field(default_factory=list, description="Specific factual claims extracted by the LLM that could be checked.") # Use default_factory
    knowledge_graph_insights: Optional[str] = Field(None, description="Insights from the knowledge graph (e.g., entity history).")
    related_url_analysis: Optional[UrlAnalysisResponse] = Field(None, description="Analysis results if a relevant URL was present in the input.")


class StatusResponse(BaseModel):
    status: str = "OK"
    version: str = Field(default="1.0.0", description="API version")
    rag_index_status: str = Field(..., description="Status of the RAG vector index.")
    kg_status: str = Field(..., description="Status of the Knowledge Graph component.")
    classifier_status: str = Field(..., description="Status of the Intent Classifier model.")

class ErrorDetail(BaseModel):
    request_id: Optional[str] = None # Make optional for easier exception raising
    error: str
    message: str

class ErrorResponse(BaseModel):
     detail: ErrorDetail
