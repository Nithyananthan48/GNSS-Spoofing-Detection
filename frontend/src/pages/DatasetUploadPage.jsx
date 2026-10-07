import React, { useState, useRef } from 'react'
import { datasetAPI } from '../services/api'
import { Upload, FileText, CheckCircle, XCircle, Info, Download, ArrowRight, HelpCircle, Table } from 'lucide-react'

export default function DatasetUploadPage() {
  const [file, setFile] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState(null)
  const [errorInfo, setErrorInfo] = useState(null)
  const [selectedMappingCol, setSelectedMappingCol] = useState('')
  const [downloadingTemplate, setDownloadingTemplate] = useState(false)
  const fileRef = useRef()

  const handleFile = (f) => {
    if (!f) return
    if (!f.name.endsWith('.csv')) {
      setErrorInfo({ message: 'Only CSV files (.csv) are supported.' })
      return
    }
    setFile(f)
    setErrorInfo(null)
    setResult(null)
    setSelectedMappingCol('')
  }

  const handleUpload = async (overrideTargetCol = null) => {
    if (!file) return
    setUploading(true)
    setErrorInfo(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)
    const targetToUse = overrideTargetCol || selectedMappingCol
    if (targetToUse) {
      formData.append('target_column', targetToUse)
    }

    try {
      const res = await datasetAPI.upload(formData)
      setResult(res.data)
      setErrorInfo(null)
    } catch (err) {
      const detail = err.response?.data?.detail
      if (typeof detail === 'object' && detail !== null) {
        setErrorInfo(detail)
        if (detail.detected_candidates && detail.detected_candidates.length > 0) {
          setSelectedMappingCol(detail.detected_candidates[0])
        } else if (detail.columns && detail.columns.length > 0) {
          setSelectedMappingCol(detail.columns[detail.columns.length - 1])
        }
      } else {
        setErrorInfo({ message: detail || 'Dataset upload and validation failed.' })
      }
    } finally {
      setUploading(false)
    }
  }

  const handleDownloadTemplate = async () => {
    setDownloadingTemplate(true)
    try {
      const res = await datasetAPI.downloadTemplate()
      const url = window.URL.createObjectURL(new Blob([res.data], { type: 'text/csv' }))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', 'gnss_sample_template.csv')
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      alert('Failed to download template. Please check backend connection.')
    } finally {
      setDownloadingTemplate(false)
    }
  }

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 16 }}>
        <div>
          <h2>📁 Upload & Validate Dataset</h2>
          <p>Upload a GNSS CSV dataset for Bi-GRU model training. The dataset must contain binary labels (0 = Normal, 1 = Spoofed).</p>
        </div>
        <button
          className="btn btn-outline"
          onClick={handleDownloadTemplate}
          disabled={downloadingTemplate}
          style={{ display: 'flex', alignItems: 'center', gap: 8 }}
        >
          <Download size={16} color="var(--accent-cyan)" />
          {downloadingTemplate ? 'Downloading...' : 'Download Sample CSV Template'}
        </button>
      </div>

      {/* Target Definition Banner */}
      <div className="alert alert-info" style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
          <HelpCircle size={20} color="var(--accent-cyan)" style={{ flexShrink: 0, marginTop: 2 }} />
          <div style={{ fontSize: '0.88rem', lineHeight: 1.6 }}>
            <strong>Target Variable Specification:</strong> The Bi-GRU network requires a target column named <code>label</code>.
            <div style={{ marginTop: 4, display: 'flex', gap: 16, flexWrap: 'wrap' }}>
              <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>🟢 label = 0 → Normal (Authentic Satellite Signal)</span>
              <span style={{ color: 'var(--accent-red)', fontWeight: 600 }}>🔴 label = 1 → Spoofed (Counterfeit Attack Signal)</span>
            </div>
            <div style={{ marginTop: 6, fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              If your dataset uses another column name (such as <code>class</code>, <code>target</code>, <code>attack</code>, or <code>spoofing</code>), the system detects it and lets you map it safely. Actual user labels are never randomly generated or altered.
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 24 }}>
        {/* Upload Column */}
        <div>
          <div
            className={`upload-area ${dragOver ? 'drag-over' : ''}`}
            onClick={() => fileRef.current.click()}
            onDragOver={e => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={e => {
              e.preventDefault()
              setDragOver(false)
              handleFile(e.dataTransfer.files[0])
            }}
          >
            <Upload size={40} color="var(--accent-blue)" style={{ margin: '0 auto', display: 'block' }} />
            <h3>Drop GNSS CSV file here or click to browse</h3>
            <p>Maximum file size: 50 MB · Standard CSV format only</p>
            {file && (
              <div style={{ marginTop: 16, padding: '10px 16px', background: 'rgba(59,130,246,0.1)', borderRadius: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
                <FileText size={16} color="var(--accent-blue)" />
                <span style={{ fontSize: '0.875rem', color: 'var(--text-primary)', fontWeight: 600 }}>{file.name}</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginLeft: 'auto' }}>
                  {(file.size / 1024).toFixed(1)} KB
                </span>
              </div>
            )}
          </div>
          <input ref={fileRef} type="file" accept=".csv" style={{ display: 'none' }} onChange={e => handleFile(e.target.files[0])} />

          <button
            className="btn btn-primary w-full btn-lg"
            style={{ marginTop: 16 }}
            onClick={() => handleUpload()}
            disabled={!file || uploading}
          >
            {uploading ? <><span className="loading-spinner" /> Validating & Ingesting Dataset...</> : <><Upload size={16} /> Upload & Validate Dataset</>}
          </button>

          {/* Detailed Error & Column Mapping Card */}
          {errorInfo && (
            <div className="card" style={{ borderColor: 'var(--accent-red)', marginTop: 16, background: 'rgba(239, 68, 68, 0.04)' }}>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10, marginBottom: 12 }}>
                <XCircle size={20} color="var(--accent-red)" style={{ flexShrink: 0, marginTop: 2 }} />
                <div>
                  <strong style={{ color: 'var(--accent-red)', fontSize: '0.95rem' }}>Dataset Validation Error</strong>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-primary)', marginTop: 4, lineHeight: 1.5 }}>
                    {errorInfo.message || errorInfo.detail || 'Validation failed.'}
                  </div>
                </div>
              </div>

              {/* Show Dataset Shape & Columns if available in error */}
              {errorInfo.shape && (
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 12, padding: '8px 12px', background: 'var(--bg-primary)', borderRadius: 6 }}>
                  <div><strong>Detected Shape:</strong> {errorInfo.shape[0]?.toLocaleString()} rows × {errorInfo.shape[1]} columns</div>
                  {errorInfo.columns && (
                    <div style={{ marginTop: 4 }}>
                      <strong>Detected Columns:</strong> {errorInfo.columns.join(', ')}
                    </div>
                  )}
                </div>
              )}

              {/* Column Mapping Selector (Requirement 8) */}
              {(errorInfo.detected_candidates || errorInfo.columns) && (
                <div style={{ marginTop: 12, padding: 14, background: 'rgba(6, 182, 212, 0.08)', borderRadius: 8, border: '1px solid rgba(6, 182, 212, 0.2)' }}>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-cyan)', marginBottom: 8 }}>
                    🎯 Map Alternative Target Column to 'label'
                  </div>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 10 }}>
                    Select which column in your CSV contains the attack classifications (0/Normal, 1/Spoofed):
                  </p>
                  <div style={{ display: 'flex', gap: 10 }}>
                    <select
                      className="form-input"
                      value={selectedMappingCol}
                      onChange={e => setSelectedMappingCol(e.target.value)}
                      style={{ flex: 1, fontSize: '0.85rem' }}
                    >
                      <option value="">-- Choose Column to Map --</option>
                      {(errorInfo.columns || errorInfo.detected_candidates || []).map(c => (
                        <option key={c} value={c}>
                          {c} {errorInfo.detected_candidates?.includes(c) ? '(Suggested Candidate)' : ''}
                        </option>
                      ))}
                    </select>
                    <button
                      className="btn btn-primary btn-sm"
                      onClick={() => handleUpload(selectedMappingCol)}
                      disabled={!selectedMappingCol || uploading}
                      style={{ whiteSpace: 'nowrap' }}
                    >
                      <ArrowRight size={14} /> Map & Re-Validate
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Result & Dataset Metadata Panel */}
        <div>
          {result ? (
            <div className="card" style={{ borderColor: 'var(--accent-green)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
                <CheckCircle size={22} color="var(--accent-green)" />
                <div>
                  <div style={{ fontWeight: 700, color: 'var(--accent-green)', fontSize: '1rem' }}>Dataset Validated & Ingested</div>
                  <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Ready for Bi-GRU temporal sequence training</div>
                </div>
              </div>

              {result.mapped_target_column && (
                <div style={{ marginBottom: 12, padding: '6px 12px', background: 'rgba(6,182,212,0.1)', borderRadius: 6, fontSize: '0.8rem', color: 'var(--accent-cyan)' }}>
                  Mapped column <strong>'{result.mapped_target_column}'</strong> → <strong>'label'</strong>
                </div>
              )}

              {/* Dataset Shape & General Details (Requirement 5) */}
              <InfoRow label="Filename" value={result.filename} />
              <InfoRow label="Dataset ID" value={`#${result.dataset_id}`} />
              <InfoRow label="Dataset Shape" value={`${result.shape ? result.shape[0]?.toLocaleString() : result.rows_count?.toLocaleString()} rows × ${result.shape ? result.shape[1] : result.features_count} columns`} />
              <InfoRow label="GNSS Observables Detected" value={`${result.available_gnss_features?.length || 0} columns`} />

              <hr className="section-divider" />

              {/* Label Distribution */}
              <div style={{ marginBottom: 16 }}>
                <div className="card-title" style={{ fontSize: '0.88rem' }}>Verified Label Distribution</div>
                {Object.entries(result.label_distribution || {}).map(([lbl, count]) => (
                  <div key={lbl} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', fontSize: '0.875rem' }}>
                    <span className={`badge ${lbl === '0' || lbl === 0 ? 'badge-normal' : 'badge-spoofed'}`}>
                      {lbl === '0' || lbl === 0 ? '🟢 0 = Normal (Authentic)' : '🔴 1 = Spoofed (Attack)'}
                    </span>
                    <span style={{ fontWeight: 700 }}>{count?.toLocaleString()} samples ({((count / result.rows_count) * 100).toFixed(1)}%)</span>
                  </div>
                ))}
              </div>

              {/* Detected Columns List */}
              <div style={{ marginBottom: 16 }}>
                <div className="card-title" style={{ fontSize: '0.88rem' }}>Detected Dataset Columns ({result.columns?.length})</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, maxHeight: 110, overflowY: 'auto', padding: 4 }}>
                  {(result.columns || []).map(col => (
                    <span
                      key={col}
                      className={`badge ${col === 'label' ? 'badge-spoofed' : result.available_gnss_features?.includes(col) ? 'badge-blue' : ''}`}
                      style={{ fontSize: '0.75rem' }}
                    >
                      {col} {col === 'label' ? '(target)' : ''}
                    </span>
                  ))}
                </div>
              </div>

              {/* Sample Data Preview Table (Requirement 5) */}
              {result.sample_preview && result.sample_preview.length > 0 && (
                <div>
                  <div className="card-title" style={{ fontSize: '0.88rem', display: 'flex', alignItems: 'center', gap: 6 }}>
                    <Table size={14} /> Sample Observations Preview (First 5 Rows)
                  </div>
                  <div style={{ overflowX: 'auto', maxHeight: 150, fontSize: '0.75rem', background: 'var(--bg-primary)', borderRadius: 6, border: '1px solid var(--border-color)' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                      <thead>
                        <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                          {Object.keys(result.sample_preview[0] || {}).slice(0, 7).map(k => (
                            <th key={k} style={{ padding: '4px 8px', textAlign: 'left', color: 'var(--text-muted)', fontSize: '0.72rem' }}>{k}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {result.sample_preview.map((row, idx) => (
                          <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                            {Object.entries(row).slice(0, 7).map(([k, val], i) => (
                              <td key={i} style={{ padding: '4px 8px', whiteSpace: 'nowrap' }}>
                                {k === 'label' ? (
                                  <span className={`badge ${val === 1 ? 'badge-spoofed' : 'badge-normal'}`} style={{ padding: '1px 6px', fontSize: '0.7rem' }}>
                                    {val}
                                  </span>
                                ) : (
                                  typeof val === 'number' ? val.toFixed(2) : String(val)
                                )}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="card" style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', minHeight: 340 }}>
              <Info size={48} style={{ opacity: 0.25, marginBottom: 16 }} />
              <p style={{ textAlign: 'center', fontSize: '0.9rem' }}>Upload a CSV file to inspect detected columns, shape, and label distribution.</p>
            </div>
          )}
        </div>
      </div>

      {/* Expected Format & Documentation (Requirements 9 & 10) */}
      <div className="card">
        <div className="card-title">Expected GNSS Dataset Format & Specification</div>
        <div style={{ overflowX: 'auto', background: 'var(--bg-primary)', borderRadius: 6, padding: 16, fontFamily: 'var(--font-mono)', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
          <pre>{`timestamp,pseudorange,carrier_phase,cn0,doppler,elevation,azimuth,latitude,longitude,altitude,velocity,label
1710000000.0,21450123.4,112717411.2,42.5,-2.1,65.4,134.2,28.61390,77.20900,216.0,0.12,0
1710000001.0,21450125.1,112717420.1,42.1,-2.3,65.4,134.3,28.61391,77.20901,216.1,0.15,0
1710000002.0,21450127.8,112717434.3,42.4,-2.0,65.5,134.4,28.61390,77.20900,215.9,0.11,0
1710000010.0,21450620.0,112720021.5,28.3,42.6,65.5,135.8,28.61850,77.21320,229.4,4.85,1
1710000011.0,21450655.2,112720208.9,27.9,45.1,65.6,136.0,28.61920,77.21410,233.1,5.12,1`}</pre>
        </div>
        <div style={{ marginTop: 14, fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.7 }}>
          <div><strong style={{ color: 'var(--text-primary)' }}>Columns Breakdown:</strong></div>
          <ul style={{ paddingLeft: 20, marginTop: 4 }}>
            <li><code>timestamp</code>: Sequential Unix epoch time (used to maintain continuous temporal ordering).</li>
            <li><code>pseudorange</code>, <code>carrier_phase</code>, <code>cn0</code>, <code>doppler</code>: Raw GNSS satellite tracking observables.</li>
            <li><code>latitude</code>, <code>longitude</code>, <code>altitude</code>, <code>velocity</code>: Computed PVT receiver coordinates.</li>
            <li><code>label</code>: <strong>Target binary ground truth</strong> (<code>0 = Normal</code>, <code>1 = Spoofed</code>).</li>
          </ul>
        </div>
      </div>
    </div>
  )
}

function InfoRow({ label, value }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '7px 0', borderBottom: '1px solid var(--border-color)', fontSize: '0.875rem' }}>
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{value}</span>
    </div>
  )
}
