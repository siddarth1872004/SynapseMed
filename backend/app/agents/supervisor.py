import time
import logging
from typing import Dict, Any, Generator, List, Optional
from datetime import datetime, timezone

from app.schemas import DiagnosticReport, PatientMetadata, DocumentExtraction, VisionFindings, RetrievalAnalysis
from app.agents.ingestion import run_document_ingestion
from app.agents.vision import run_vision_inference
from app.agents.retrieval import run_guidelines_retrieval

logger = logging.getLogger("supervisor_agent")

class SupervisorOrchestrator:
    """
    Supervisor Agent that coordinates routing, execution, and synthesis of
    diagnostic and research tasks. Operates as a stateful agent system.
    """
    def __init__(self, state: Dict[str, Any]):
        self.state = {
            "patient_id": state.get("patient_id", "PAT-UNKNOWN"),
            "age": state.get("age"),
            "gender": state.get("gender"),
            "document_path": state.get("document_path"),
            "image_path": state.get("image_path"),
            "extracted_text": None,
            "symptoms": [],
            "vision_results": None,
            "retrieval_results": None,
            "logs": [],
            "next_step": "INGESTION",
            "is_complete": False
        }

    def log(self, message: str) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = f"[{timestamp}] {message}"
        self.state["logs"].append(log_entry)
        logger.info(log_entry)
        return {"step": self.state["next_step"], "log": message, "timestamp": timestamp}

    def execute_step(self) -> Generator[Dict[str, Any], None, None]:
        """
        Executes the supervisor routing loop step-by-step, yielding progress states.
        This provides streaming execution tracking for the client UI.
        """
        yield self.log(f"Supervisor: Initiated Diagnostic Copilot run for Patient {self.state['patient_id']}")
        
        # --- STEP 1: Document Ingestion Route ---
        self.state["next_step"] = "INGESTION"
        if self.state["document_path"]:
            yield self.log("Supervisor Routing: Document uploaded. Delegating task to Document Ingestion Agent...")
            # Simulate slight processing delay for realism in streams
            time.sleep(0.8)
            
            ingest_res = run_document_ingestion(self.state["document_path"])
            self.state["extracted_text"] = ingest_res["extracted_text"]
            self.state["symptoms"] = ingest_res["symptoms"]
            
            yield self.log(f"Document Ingestion Agent: Complete. {ingest_res['log']}")
            yield self.log(f"Document Ingestion Agent: Extracted Symptoms: {self.state['symptoms']}")
        else:
            yield self.log("Supervisor Routing: No text document uploaded. Ingestion step skipped.")

        # --- STEP 2: Vision Inference Route ---
        self.state["next_step"] = "VISION"
        if self.state["image_path"]:
            yield self.log("Supervisor Routing: MRI/X-ray scan uploaded. Delegating task to Vision Inference Agent...")
            time.sleep(1.0)
            
            vision_res = run_vision_inference(self.state["image_path"])
            self.state["vision_results"] = vision_res
            
            yield self.log(f"Vision Inference Agent: Complete. {vision_res['log']}")
            if vision_res["has_finding"]:
                yield self.log(
                    f"Vision Inference Agent: Detected {vision_res['finding_type']} "
                    f"({vision_res['tumor_size_mm']} mm, confidence: {vision_res['confidence']*100:.1f}%)"
                )
            else:
                yield self.log("Vision Inference Agent: Scan cleared. No abnormal mass detected.")
        else:
            yield self.log("Supervisor Routing: No medical scan uploaded. Vision step skipped.")

        # --- STEP 3: Knowledge Retrieval Route ---
        self.state["next_step"] = "RETRIEVAL"
        yield self.log("Supervisor Routing: Queries formulated. Delegating task to RAG Retrieval Agent...")
        time.sleep(0.8)
        
        # Perform retrieval based on findings
        finding = self.state["vision_results"]["finding_type"] if self.state["vision_results"] else "Brain Tumor"
        retrieval_res = run_guidelines_retrieval(self.state["symptoms"], finding)
        self.state["retrieval_results"] = retrieval_res
        
        yield self.log(f"Retrieval Agent: Complete. {retrieval_res['log']}")
        for match in retrieval_res["matched_guidelines"]:
            yield self.log(f"Retrieval Agent: Referenced guidelines: {match['title']} ({match['author_or_org']})")

        # --- STEP 4: Report Synthesis & Verification ---
        self.state["next_step"] = "SYNTHESIS"
        yield self.log("Supervisor Routing: Synthesizing clinical inputs, vision segmentation, and research literature...")
        time.sleep(1.0)
        
        # Build synthesis
        report = self.synthesize_report()
        self.state["is_complete"] = True
        self.state["next_step"] = "COMPLETED"
        
        yield self.log("Supervisor: Diagnostic report compiled and Pydantic schema validated.")
        yield {"step": "COMPLETED", "report": report.model_dump(), "timestamp": datetime.now(timezone.utc).isoformat()}

    def synthesize_report(self) -> DiagnosticReport:
        """
        Synthesizes worker agent logs into a strictly validated DiagnosticReport Pydantic schema.
        """
        patient_metadata = PatientMetadata(
            patient_id=self.state["patient_id"],
            age=self.state["age"],
            gender=self.state["gender"]
        )
        
        doc_extraction = None
        if self.state["document_path"] and self.state["extracted_text"]:
            # Simple summarization helper
            summary = self.state["extracted_text"][:200] + "..." if len(self.state["extracted_text"]) > 200 else self.state["extracted_text"]
            doc_extraction = DocumentExtraction(
                extracted_text_summary=summary,
                detected_symptoms=self.state["symptoms"],
                clinical_history_notes=self.state["extracted_text"]
            )
            
        vision_findings = None
        if self.state["vision_results"]:
            res = self.state["vision_results"]
            vision_findings = VisionFindings(
                has_finding=res["has_finding"],
                finding_type=res["finding_type"],
                tumor_size_mm=res["tumor_size_mm"] if res["has_finding"] else None,
                confidence=res["confidence"],
                coordinates=res["coordinates"]
            )
            
        retrieval_analysis = None
        if self.state["retrieval_results"]:
            res = self.state["retrieval_results"]
            retrieval_analysis = RetrievalAnalysis(
                matched_guidelines=res["matched_guidelines"],
                is_grounded=res["is_grounded"]
            )
            
        # Draft synthetic clinical overview text based on worker outputs
        findings_str = ""
        grading = "Routine"
        actions = ["Routine neurological follow-up as indicated."]
        
        if vision_findings and vision_findings.has_finding:
            findings_str = (
                f"Neuroimaging reveals a localized tissue mass consistent with {vision_findings.finding_type}, "
                f"measuring approximately {vision_findings.tumor_size_mm} mm. "
            )
            if vision_findings.tumor_size_mm > 20.0 or "Glioma" in vision_findings.finding_type:
                grading = "Critical"
                actions = [
                    "Urgent neurosurgical consultation for resection planning.",
                    "Obtain multi-sequence brain MRI with gadolinium contrast.",
                    "Initiate corticosteroid therapy (Dexamethasone) for edema control if indicated.",
                    "Refer to neuro-oncology board review."
                ]
            else:
                grading = "Urgent"
                actions = [
                    "Neurosurgical consult for evaluation.",
                    "Obtain follow-up brain MRI in 3 months to assess growth kinetics.",
                    "Consult endocrinology for hormone levels (if pituitary-related)."
                ]
        else:
            findings_str = "No focal mass effect, intracranial hemorrhage, or vasogenic edema identified on imaging. "
            
        history_str = ""
        if doc_extraction and doc_extraction.detected_symptoms:
            history_str = f"Patient history reports clinical symptoms of: {', '.join(doc_extraction.detected_symptoms)}. "
            
        summary_text = (
            f"Preliminary Assessment: {findings_str}{history_str} findings have been correlated "
            "with localized guidelines. Recommend formal clinical review by attending radiologist and physician."
        )
        
        return DiagnosticReport(
            patient_metadata=patient_metadata,
            document_extraction=doc_extraction,
            vision_findings=vision_findings,
            retrieval_analysis=retrieval_analysis,
            synthesized_diagnostic_summary=summary_text,
            recommended_follow_ups=actions,
            grading_level=grading,
            generated_at=datetime.now(timezone.utc).isoformat()
        )
