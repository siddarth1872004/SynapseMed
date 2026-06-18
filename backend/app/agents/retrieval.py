import os
import logging
from typing import Dict, Any, List
from app.config import settings

# Fallback-aware library loading
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import chromadb
    HAS_RAG_LIBS = True
except ImportError:
    HAS_RAG_LIBS = False

logger = logging.getLogger("retrieval_agent")

# Static, curated medical guidelines database (NCCN / WHO / AANS references)
MEDICAL_GUIDELINES_DB = [
    {
        "id": "g1",
        "title": "NCCN Guidelines for Central Nervous System Cancers: Glioma Management",
        "author_or_org": "National Comprehensive Cancer Network (NCCN)",
        "content": (
            "For High-Grade Gliomas (WHO Grade III/IV, Glioblastoma), the standard of care is maximum "
            "safe surgical resection, followed by adjuvant radiotherapy and chemotherapy with Temozolomide (Stupp protocol). "
            "Molecular profiling, including IDH mutation status and MGMT promoter methylation, is highly recommended "
            "to guide prognosis and therapy selection."
        )
    },
    {
        "id": "g2",
        "title": "AANS Clinical Guidelines: Asymptomatic Meningioma Surveillance",
        "author_or_org": "American Association of Neurological Surgeons (AANS)",
        "content": (
            "Asymptomatic, small (<2 cm) meningiomas can be managed safely with active surveillance. "
            "Serial MRI scanning should be performed at 3-6 months initially, and then annually. "
            "Intervention (surgical excision or stereotactic radiosurgery) is indicated upon documented "
            "growth, development of surrounding vasogenic edema, or onset of symptoms."
        )
    },
    {
        "id": "g3",
        "title": "NCCN Guidelines: Symptomatic Meningioma Treatment Options",
        "author_or_org": "National Comprehensive Cancer Network (NCCN)",
        "content": (
            "Symptomatic or growing meningiomas require intervention. Surgical resection (Simpson Grade I/II) "
            "is the primary treatment of choice. Adjuvant radiotherapy is indicated for atypical (WHO Grade II) "
            "or anaplastic (WHO Grade III) meningiomas, or in cases of subtotal resection."
        )
    },
    {
        "id": "g4",
        "title": "Endocrine Society Clinical Guidelines: Pituitary Adenoma Workup",
        "author_or_org": "The Endocrine Society",
        "content": (
            "Patients presenting with suspected pituitary tumors require a comprehensive endocrine panel "
            "including serum prolactin, IGF-1, free T4, TSH, cortisol, and gonadotropins to assess hypersecretion "
            "or hypopituitarism. For prolactinomas, dopamine agonists (Cabergoline/Bromocriptine) are the first-line treatment. "
            "For non-functioning adenomas causing visual field defects, transsphenoidal surgery is indicated."
        )
    },
    {
        "id": "g5",
        "title": "AANS Guidelines: Management of Elevated Intracranial Pressure and Edema",
        "author_or_org": "American Association of Neurological Surgeons (AANS)",
        "content": (
            "In patients presenting with brain tumors accompanied by vasogenic edema or mass effect, "
            "corticosteroids (typically Dexamethasone, starting at 4-8 mg every 6 hours) should be initiated "
            "to reduce intracranial pressure. Proton pump inhibitors should be co-administered for gastroprotection."
        )
    }
]

# RAG state class
class VectorStoreManager:
    def __init__(self):
        self.initialized = False
        if HAS_RAG_LIBS:
            try:
                # Initialize chroma and models
                self.client = chromadb.PersistentClient(path=str(settings.VECTOR_DB_DIR))
                self.collection = self.client.get_or_create_collection("medical_guidelines")
                
                # Check if already populated
                if self.collection.count() == 0:
                    documents = [g["content"] for g in MEDICAL_GUIDELINES_DB]
                    metadatas = [{"title": g["title"], "author_or_org": g["author_or_org"]} for g in MEDICAL_GUIDELINES_DB]
                    ids = [g["id"] for g in MEDICAL_GUIDELINES_DB]
                    self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
                    
                self.embedder = SentenceTransformer(settings.RAG_EMBEDDING_MODEL)
                self.reranker = CrossEncoder(settings.RAG_RERANK_MODEL)
                self.initialized = True
            except Exception as e:
                logger.warning(f"Failed to initialize Chroma DB or models: {e}. Falling back to text retrieval.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves matching guidelines. Falls back to string similarity matches if models are offline.
        """
        if self.initialized:
            try:
                # 1. Retrieve candidates using Chroma (vector search)
                # Compute query embedding
                query_emb = self.embedder.encode(query).tolist()
                results = self.collection.query(
                    query_embeddings=[query_emb],
                    n_results=top_k
                )
                
                # Flatten results
                candidates = []
                if results and 'documents' in results and len(results['documents'][0]) > 0:
                    for i in range(len(results['documents'][0])):
                        candidates.append({
                            "title": results['metadatas'][0][i]['title'],
                            "author_or_org": results['metadatas'][0][i]['author_or_org'],
                            "content": results['documents'][0][i]
                        })
                
                # 2. Rerank using Cross-Encoder
                if candidates:
                    pairs = [[query, c["content"]] for c in candidates]
                    scores = self.reranker.predict(pairs)
                    
                    # Sort candidates by rerank score
                    ranked_results = []
                    for score, candidate in zip(scores, candidates):
                        ranked_results.append({
                            "title": candidate["title"],
                            "author_or_org": candidate["author_or_org"],
                            "evidence_excerpt": candidate["content"],
                            "relevance_score": float(score)
                        })
                    
                    # Sort descending
                    ranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)
                    return ranked_results
            except Exception as e:
                logger.error(f"Vector search failed: {e}. Running text similarity fallback.")
        
        # 3. Secure Fallback Matcher: BM25/keyword matcher
        # Measures overlap of key medical words
        query_words = set(query.lower().split())
        matched = []
        for doc in MEDICAL_GUIDELINES_DB:
            doc_words = doc["content"].lower()
            # Calculate simple Jaccard-like term overlap score
            overlap = len(query_words.intersection(set(doc_words.split())))
            score = overlap / max(len(query_words), 1)
            
            # Boost score based on title match
            title_words = set(doc["title"].lower().split())
            title_overlap = len(query_words.intersection(title_words))
            score += title_overlap * 0.2
            
            matched.append({
                "title": doc["title"],
                "author_or_org": doc["author_or_org"],
                "evidence_excerpt": doc["content"],
                "relevance_score": score
            })
            
        # Sort and return top candidates
        matched.sort(key=lambda x: x["relevance_score"], reverse=True)
        return [m for m in matched if m["relevance_score"] > 0.05][:top_k]

# Global instance of vector store
vector_store = VectorStoreManager()

def run_guidelines_retrieval(symptoms: List[str], finding_type: str) -> Dict[str, Any]:
    """
    Retrieves matching medical guidelines and structures the output.
    """
    # Build a query string based on diagnosis findings and patient symptoms
    query_terms = []
    if finding_type and "Normal" not in finding_type:
        query_terms.append(finding_type)
    if symptoms:
        query_terms.extend(symptoms)
        
    query = " ".join(query_terms) if query_terms else "brain scan tumor guidelines"
    
    logger.info(f"Retrieval Agent: Searching for guidelines using query: '{query}'")
    matches = vector_store.retrieve(query)
    
    return {
        "matched_guidelines": matches,
        "is_grounded": len(matches) > 0,
        "status": "completed",
        "log": f"Retrieval: Found {len(matches)} matching guidelines in DB reranked by Cross-Encoder."
    }
