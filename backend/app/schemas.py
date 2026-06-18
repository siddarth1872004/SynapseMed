from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class PatientMetadata(BaseModel):
    patient_id: str = Field(..., description="Unique anonymized patient identifier")
    age: Optional[int] = Field(None, ge=0, le=125, description="Patient age")
    gender: Optional[str] = Field(None, description="Patient gender (e.g., M, F, Other)")

class DocumentExtraction(BaseModel):
    extracted_text_summary: str = Field(..., description="Summarized text parsed from scanned notes")
    detected_symptoms: List[str] = Field(default_factory=list, description="List of symptoms detected in history notes")
    clinical_history_notes: str = Field(..., description="Cleaned, structured clinical history text")

class VisionFindings(BaseModel):
    has_finding: bool = Field(..., description="Flag indicating if any anomaly/tumor was detected")
    finding_type: str = Field(..., description="Classification category (e.g. Glioma, Meningioma, Pituitary, Normal, No finding)")
    tumor_size_mm: Optional[float] = Field(None, description="Estimated tumor diameter in millimeters")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model prediction confidence score")
    coordinates: Optional[Dict[str, float]] = Field(None, description="Detected bounding box center or area (x, y, w, h)")

class RetrievalSource(BaseModel):
    title: str = Field(..., description="Title of the medical guideline or publication")
    author_or_org: str = Field(..., description="Publishing body or author (e.g., WHO, NCCN)")
    evidence_excerpt: str = Field(..., description="Specific grounding text excerpt matching patient case")
    relevance_score: float = Field(..., description="Matching relevance/similarity score")

class RetrievalAnalysis(BaseModel):
    matched_guidelines: List[RetrievalSource] = Field(default_factory=list, description="Guidelines used to ground findings")
    is_grounded: bool = Field(..., description="True if recommendations strictly match medical standards")

class DiagnosticReport(BaseModel):
    patient_metadata: PatientMetadata
    document_extraction: Optional[DocumentExtraction] = None
    vision_findings: Optional[VisionFindings] = None
    retrieval_analysis: Optional[RetrievalAnalysis] = None
    synthesized_diagnostic_summary: str = Field(..., description="Final synthesized diagnostic draft summary")
    recommended_follow_ups: List[str] = Field(default_factory=list, description="List of recommended clinical next-steps")
    grading_level: str = Field(..., description="Urgency / Severity grade (e.g., Critical, Urgent, Routine)")
    generated_at: str = Field(..., description="ISO timestamp of report generation")
