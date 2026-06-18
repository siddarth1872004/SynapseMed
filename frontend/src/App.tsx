import React, { useState, useRef, useEffect } from 'react';

interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface Guideline {
  title: string;
  author_or_org: string;
  evidence_excerpt: string;
  relevance_score: number;
}

interface Report {
  patient_metadata: {
    patient_id: string;
    age?: number;
    gender?: string;
  };
  document_extraction?: {
    extracted_text_summary: string;
    detected_symptoms: string[];
    clinical_history_notes: string;
  };
  vision_findings?: {
    has_finding: boolean;
    finding_type: string;
    tumor_size_mm?: number;
    confidence: number;
    coordinates?: BBox;
  };
  retrieval_analysis?: {
    matched_guidelines: Guideline[];
    is_grounded: boolean;
  };
  synthesized_diagnostic_summary: string;
  recommended_follow_ups: string[];
  grading_level: string;
  generated_at: string;
}

function App() {
  // Form State
  const [patientId, setPatientId] = useState<string>('PAT-4829');
  const [age, setAge] = useState<string>('42');
  const [gender, setGender] = useState<string>('F');
  
  // Files State
  const [docFile, setDocFile] = useState<File | null>(null);
  const [imgFile, setImgFile] = useState<File | null>(null);
  const [imgPreview, setImgPreview] = useState<string | null>(null);
  
  // Execution State
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [activeStep, setActiveStep] = useState<string>('IDLE'); // IDLE, INGESTION, VISION, RETRIEVAL, SYNTHESIS, COMPLETED, ERROR
  const [logs, setLogs] = useState<string[]>([]);
  const [report, setReport] = useState<Report | null>(null);
  
  // Bounding box overlay tracking
  const imgRef = useRef<HTMLImageElement>(null);
  const [bboxStyle, setBboxStyle] = useState<React.CSSProperties>({});

  // Clean preview URL on unmount
  useEffect(() => {
    return () => {
      if (imgPreview) URL.revokeObjectURL(imgPreview);
    };
  }, [imgPreview]);

  // Update bounding box coordinates relative to rendered image dimensions
  useEffect(() => {
    if (report?.vision_findings?.coordinates && imgRef.current) {
      const img = imgRef.current;
      const coords = report.vision_findings.coordinates;
      
      const updateBBox = () => {
        const renderedWidth = img.clientWidth;
        const renderedHeight = img.clientHeight;
        const naturalWidth = img.naturalWidth || 512;
        const naturalHeight = img.naturalHeight || 512;
        
        // Calculate scaling ratios
        const scaleX = renderedWidth / naturalWidth;
        const scaleY = renderedHeight / naturalHeight;
        
        // Compute overlay positions
        // x, y are centers, w, h are width, height in natural pixels
        const width = coords.w * scaleX;
        const height = coords.h * scaleY;
        const left = (coords.x - coords.w / 2) * scaleX;
        const top = (coords.y - coords.h / 2) * scaleY;
        
        setBboxStyle({
          left: `${left}px`,
          top: `${top}px`,
          width: `${width}px`,
          height: `${height}px`,
          display: 'block'
        });
      };
      
      // Update when image loads
      if (img.complete) {
        updateBBox();
      } else {
        img.onload = updateBBox;
      }
      
      // Update on resize
      window.addEventListener('resize', updateBBox);
      return () => window.removeEventListener('resize', updateBBox);
    } else {
      setBboxStyle({ display: 'none' });
    }
  }, [report, imgPreview]);

  // Handle files
  const handleDocChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setDocFile(e.target.files[0]);
    }
  };

  const handleImgChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImgFile(file);
      if (imgPreview) URL.revokeObjectURL(imgPreview);
      setImgPreview(URL.createObjectURL(file));
    }
  };

  const clearFiles = () => {
    setDocFile(null);
    setImgFile(null);
    if (imgPreview) URL.revokeObjectURL(imgPreview);
    setImgPreview(null);
    setReport(null);
    setActiveStep('IDLE');
    setLogs([]);
  };

  // Launch analysis and read SSE stream
  const launchAnalysis = async () => {
    if (!patientId) return;
    
    setIsRunning(true);
    setReport(null);
    setLogs([]);
    setActiveStep('INIT');
    
    const formData = new FormData();
    formData.append('patient_id', patientId);
    if (age) formData.append('age', age);
    if (gender) formData.append('gender', gender);
    if (docFile) formData.append('document', docFile);
    if (imgFile) formData.append('image', imgFile);

    try {
      // Direct localhost API connection
      const response = await fetch('http://127.0.0.1:8000/api/copilot/run', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Server returned error: ${response.statusText}`);
      }

      if (!response.body) {
        throw new Error('No readable stream returned from endpoint.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        
        // Save the last partial line back to buffer
        buffer = lines.pop() || '';
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6).trim();
            if (!dataStr) continue;
            
            try {
              const data = JSON.parse(dataStr);
              
              if (data.log) {
                setLogs(prev => [...prev, data.log]);
              }
              if (data.step) {
                setActiveStep(data.step);
              }
              if (data.step === 'COMPLETED' && data.report) {
                setReport(data.report);
                setActiveStep('COMPLETED');
              }
              if (data.step === 'ERROR') {
                setActiveStep('ERROR');
              }
            } catch (err) {
              console.error('Error parsing SSE json line:', err);
            }
          }
        }
      }
    } catch (err: any) {
      setLogs(prev => [...prev, `[ERROR] Execution aborted: ${err.message}`]);
      setActiveStep('ERROR');
    } finally {
      setIsRunning(false);
    }
  };

  const getStepProgressWidth = () => {
    switch (activeStep) {
      case 'IDLE': return '0%';
      case 'INIT': return '10%';
      case 'INGESTION': return '30%';
      case 'VISION': return '55%';
      case 'RETRIEVAL': return '80%';
      case 'SYNTHESIS': return '90%';
      case 'COMPLETED': return '100%';
      default: return '0%';
    }
  };

  return (
    <div>
      {/* Header */}
      <header className="app-header">
        <div className="brand">
          <div className="brand-icon">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
            </svg>
          </div>
          <h1>Multi-Modal Diagnostic & Research Copilot</h1>
          <span>v1.0.0 (LangGraph orchestrated)</span>
        </div>
        <div>
          <span style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)' }}>
            System state: <strong style={{ color: 'var(--color-success)' }}>Active</strong>
          </span>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="workspace">
        {/* Left Control Panel */}
        <aside className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div>
            <h2 className="form-title">Patient Intake</h2>
            <div className="input-group">
              <label>Patient ID (Anonymized)</label>
              <input 
                type="text" 
                className="input-field" 
                value={patientId}
                onChange={e => setPatientId(e.target.value)}
                disabled={isRunning} 
              />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
              <div className="input-group">
                <label>Age</label>
                <input 
                  type="number" 
                  className="input-field" 
                  value={age}
                  onChange={e => setAge(e.target.value)}
                  disabled={isRunning} 
                />
              </div>
              <div className="input-group">
                <label>Biological Gender</label>
                <select 
                  className="input-field" 
                  value={gender}
                  onChange={e => setGender(e.target.value)}
                  disabled={isRunning}
                >
                  <option value="M">Male</option>
                  <option value="F">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
            </div>
          </div>

          <div>
            <h2 className="form-title">Clinical History</h2>
            <div className="upload-zone">
              <svg className="upload-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <div className="upload-text">Upload Patient History</div>
              <div className="upload-sub">PDF, TXT or DOCX (max 10MB)</div>
              <input 
                type="file" 
                accept=".pdf,.txt,.docx" 
                style={{ display: 'none' }} 
                id="doc-upload"
                onChange={handleDocChange}
                disabled={isRunning}
              />
              <button 
                type="button" 
                className="input-field" 
                style={{ width: 'auto', marginTop: '0.5rem', cursor: 'pointer' }}
                onClick={() => document.getElementById('doc-upload')?.click()}
                disabled={isRunning}
              >
                Browse File
              </button>
            </div>
            {docFile && (
              <div className="file-pill">
                <div className="file-info">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                  <span>{docFile.name}</span>
                </div>
                <button className="remove-btn" onClick={() => setDocFile(null)} disabled={isRunning}>&times;</button>
              </div>
            )}
          </div>

          <div>
            <h2 className="form-title">Medical Imaging</h2>
            <div className="upload-zone">
              <svg className="upload-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <div className="upload-text">Upload MRI / X-ray</div>
              <div className="upload-sub">PNG, JPG or TIFF (max 10MB)</div>
              <input 
                type="file" 
                accept=".png,.jpg,.jpeg,.tiff,.tif" 
                style={{ display: 'none' }} 
                id="img-upload"
                onChange={handleImgChange}
                disabled={isRunning}
              />
              <button 
                type="button" 
                className="input-field" 
                style={{ width: 'auto', marginTop: '0.5rem', cursor: 'pointer' }}
                onClick={() => document.getElementById('img-upload')?.click()}
                disabled={isRunning}
              >
                Browse Image
              </button>
            </div>
            {imgFile && (
              <div className="file-pill">
                <div className="file-info">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                  <span>{imgFile.name}</span>
                </div>
                <button className="remove-btn" onClick={() => { setImgFile(null); setImgPreview(null); }} disabled={isRunning}>&times;</button>
              </div>
            )}
          </div>

          <div style={{ marginTop: 'auto' }}>
            <button 
              className="run-button" 
              onClick={launchAnalysis} 
              disabled={isRunning || (!docFile && !imgFile)}
            >
              {isRunning ? 'Analyzing...' : 'Run Diagnostics Copilot'}
            </button>
            {(docFile || imgFile) && (
              <button 
                className="input-field" 
                style={{ marginTop: '0.5rem', background: 'transparent', border: '1px solid rgba(244,63,94,0.3)', color: 'var(--color-critical)', cursor: 'pointer' }}
                onClick={clearFiles}
                disabled={isRunning}
              >
                Clear Intake Data
              </button>
            )}
          </div>
        </aside>

        {/* Right Dashboard */}
        <section className="main-dashboard">
          {/* Agent Node Graph */}
          <div className="glass-panel agent-flow-container">
            <div className="agent-flow-line">
              <div className="agent-flow-line-progress" style={{ width: getStepProgressWidth() }} />
            </div>

            <div className={`agent-node ${activeStep === 'INIT' ? 'active' : ''} ${['INGESTION', 'VISION', 'RETRIEVAL', 'SYNTHESIS', 'COMPLETED'].includes(activeStep) ? 'completed' : ''}`}>
              <div className="node-dot">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
              </div>
              <div className="node-label">Supervisor</div>
            </div>

            <div className={`agent-node ${activeStep === 'INGESTION' ? 'active' : ''} ${['VISION', 'RETRIEVAL', 'SYNTHESIS', 'COMPLETED'].includes(activeStep) ? 'completed' : ''}`}>
              <div className="node-dot">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
              </div>
              <div className="node-label">Ingestion Agent</div>
            </div>

            <div className={`agent-node ${activeStep === 'VISION' ? 'active' : ''} ${['RETRIEVAL', 'SYNTHESIS', 'COMPLETED'].includes(activeStep) ? 'completed' : ''}`}>
              <div className="node-dot">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
              </div>
              <div className="node-label">Vision Agent</div>
            </div>

            <div className={`agent-node ${activeStep === 'RETRIEVAL' ? 'active' : ''} ${['SYNTHESIS', 'COMPLETED'].includes(activeStep) ? 'completed' : ''}`}>
              <div className="node-dot">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
              </div>
              <div className="node-label">Retrieval Agent</div>
            </div>

            <div className={`agent-node ${activeStep === 'SYNTHESIS' ? 'active' : ''} ${['COMPLETED'].includes(activeStep) ? 'completed' : ''}`}>
              <div className="node-dot">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
              </div>
              <div className="node-label">Synthesizer</div>
            </div>
          </div>

          {/* Grid: Console Logs & Diagnostic Report */}
          <div className="dashboard-grid">
            {/* Live Console Logs */}
            <div className="glass-panel console-panel">
              <div className="console-header">
                <h3 style={{ margin: 0, fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Agent Console Logs</h3>
                <div className="console-status">
                  <span className={`status-dot ${isRunning ? 'active' : ''}`} />
                  <span>{isRunning ? 'EXECUTION STREAMING' : activeStep === 'COMPLETED' ? 'RUN FINISHED' : 'READY'}</span>
                </div>
              </div>
              <div className="console-output">
                {logs.length === 0 ? (
                  <div style={{ color: 'var(--color-text-muted)', fontStyle: 'italic' }}>
                    System logs will display here during run execution...
                  </div>
                ) : (
                  logs.map((log, idx) => {
                    let logClass = "console-line";
                    if (log.includes("[ERROR]")) logClass += " error";
                    else if (log.includes("Agent: Complete")) logClass += " success";
                    else if (log.includes("Supervisor Routing")) logClass += " highlight";
                    
                    return (
                      <div key={idx} className={logClass}>
                        {log}
                      </div>
                    );
                  })
                )}
              </div>
            </div>

            {/* Structured Medical Report */}
            <div className="glass-panel report-panel">
              {!report ? (
                <div className="report-placeholder">
                  <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <div>Diagnostic report summary will construct here...</div>
                  <div style={{ fontSize: '0.75rem' }}>Upload inputs and execute the system above.</div>
                </div>
              ) : (
                <div className="report-content">
                  <div className="report-header-info">
                    <div>
                      <h4 style={{ margin: 0, fontSize: '1rem' }}>Patient Report: {report.patient_metadata.patient_id}</h4>
                      <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
                        Age: {report.patient_metadata.age || 'N/A'} | Gender: {report.patient_metadata.gender || 'N/A'}
                      </span>
                    </div>
                    <span className={`badge ${report.grading_level.toLowerCase()}`}>
                      {report.grading_level}
                    </span>
                  </div>

                  {/* Document Ingestion Results */}
                  {report.document_extraction && (
                    <div>
                      <div className="report-section-title">Clinical History Extract</div>
                      <div className="section-box" style={{ fontSize: '0.8rem', lineHeight: '1.4' }}>
                        {report.document_extraction.extracted_text_summary}
                        {report.document_extraction.detected_symptoms.length > 0 && (
                          <div style={{ marginTop: '0.4rem', color: 'var(--color-primary)', fontWeight: 500 }}>
                            Identified Symptoms: {report.document_extraction.detected_symptoms.join(', ')}
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Vision Inference Findings */}
                  {report.vision_findings && (
                    <div>
                      <div className="report-section-title">Neuro-Imaging Analysis</div>
                      <div className="section-box findings-grid">
                        <div className="metric-item">
                          <span className="metric-lbl">Diagnostic Classification</span>
                          <span className="metric-val">{report.vision_findings.finding_type}</span>
                        </div>
                        <div className="metric-item">
                          <span className="metric-lbl">Confidence Score</span>
                          <span className="metric-val">{(report.vision_findings.confidence * 100).toFixed(1)}%</span>
                        </div>
                        {report.vision_findings.tumor_size_mm !== undefined && (
                          <div className="metric-item">
                            <span className="metric-lbl">Est. Mass Size</span>
                            <span className="metric-val" style={{ color: 'var(--color-critical)' }}>
                              {report.vision_findings.tumor_size_mm} mm
                            </span>
                          </div>
                        )}
                      </div>
                      
                      {/* Render Visual Overlay of detection coordinates */}
                      {imgPreview && report.vision_findings.coordinates && (
                        <div className="scan-visualization-box">
                          <img 
                            ref={imgRef}
                            src={imgPreview} 
                            alt="Patient Brain MRI Scan" 
                            className="visualized-mri"
                          />
                          <div className="bounding-box-overlay" style={bboxStyle}>
                            <span className="bbox-label">
                              {report.vision_findings.finding_type} ({report.vision_findings.tumor_size_mm}mm)
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Grounded Guidelines Retrieval */}
                  {report.retrieval_analysis?.matched_guidelines && report.retrieval_analysis.matched_guidelines.length > 0 && (
                    <div>
                      <div className="report-section-title">Literature Retrieval & Grounding</div>
                      <div>
                        {report.retrieval_analysis.matched_guidelines.map((guide, idx) => (
                          <div key={idx} className="guideline-card">
                            <div className="guideline-title">
                              {guide.title}{' '}
                              <span style={{ fontSize: '0.7rem', color: 'var(--color-primary)', fontWeight: 'normal' }}>
                                (Rank Match: {(guide.relevance_score * 100).toFixed(0)}%)
                              </span>
                            </div>
                            <div className="guideline-excerpt">
                              &ldquo;{guide.evidence_excerpt}&rdquo;
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Summary & Recommended Follow-ups */}
                  <div>
                    <div className="report-section-title">Clinical Synthesis Report</div>
                    <div className="section-box" style={{ fontSize: '0.85rem', marginBottom: '0.75rem' }}>
                      {report.synthesized_diagnostic_summary}
                    </div>
                    {report.recommended_follow_ups.length > 0 && (
                      <div>
                        <div style={{ fontSize: '0.75rem', fontWeight: 'bold', color: 'var(--color-text-muted)', marginBottom: '0.25rem', textTransform: 'uppercase' }}>
                          Recommended Interventions
                        </div>
                        <ul className="followup-list">
                          {report.recommended_follow_ups.map((item, idx) => (
                            <li key={idx} style={{ color: report.grading_level === 'Critical' ? 'var(--color-warning)' : 'inherit' }}>
                              {item}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
