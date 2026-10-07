import React, { useState, useRef } from 'react'
import { predictionAPI } from '../services/api'
import { Radar, Upload, ShieldAlert, ShieldCheck, FileText, BarChart2 } from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from 'recharts'

export default function DetectionPage() {
  const [file, setFile] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const fileRef = useRef()

  const handleFile = (f) => {
    if (!f) return
    if (!f.name.endsWith('.csv')) { setError('Only CSV files supported.'); return }
    setFile(f); setError(''); setResult(null)
  }

  const runDetection = async () => {
    if (!file) return
    setLoading(true); setError(''); setResult(null)
    const fd = new FormData()
    fd.append('file', file)
    try {
      const res = await predictionAPI.predict(fd)
      setResult(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Detection failed.')
    } finally {
      setLoading(false)
    }
  }

  const isSpoofed = result?.overall_prediction === 'SPOOFED'

  // Chart data: distribution of confidences in buckets
  const confidenceHistData = (() => {
    if (!result?.sequence_details) return []
    const buckets = Array(10).fill(0)
    result.sequence_details.forEach(s => {
      const idx = Math.min(9, Math.floor(s.confidence / 10))
      buckets[idx]++
    })
    return buckets.map((count, i) => ({
      range: `${i * 10}-${i * 10 + 10}%`,
      count,
    }))
  })()

  return (
    <div>
      <div className="page-header">
        <h2><Radar size={22} style={{ verticalAlign: 'middle', marginRight: 8 }} />GNSS Spoofing Detection</h2>
        <p>Upload a GNSS observation CSV and run the trained Bi-GRU model to detect spoofing attacks.</p>
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
        {/* Upload Section */}
        <div>
          <div
            className={`upload-area ${dragOver ? 'drag-over' : ''}`}
            onClick={() => fileRef.current.click()}
            onDragOver={e => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={e => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]) }}
          >
            <Radar size={44} color="var(--accent-cyan)" style={{ margin: '0 auto', display: 'block' }} />
            <h3>Upload GNSS Observation CSV</h3>
            <p>Drop a CSV file to run spoofing detection</p>
            {file && (
              <div style={{ marginTop: 12, padding: '8px 14px', background: 'rgba(6,182,212,0.1)', borderRadius: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
                <FileText size={15} color="var(--accent-cyan)" />
                <span style={{ fontSize: '0.85rem' }}>{file.name}</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginLeft: 'auto' }}>{(file.size / 1024).toFixed(1)} KB</span>
              </div>
            )}
          </div>
          <input ref={fileRef} type="file" accept=".csv" style={{ display: 'none' }} onChange={e => handleFile(e.target.files[0])} />

          <button
            className="btn btn-primary w-full btn-lg"
            style={{ marginTop: 16 }}
            onClick={runDetection}
            disabled={!file || loading}
          >
            {loading ? <><span className="loading-spinner" /> Running Detection...</> : <><Radar size={16} /> Run Detection</>}
          </button>

          <div className="alert alert-info" style={{ marginTop: 16 }}>
            The Bi-GRU model must be trained before running detection. The uploaded CSV will be preprocessed using the saved scaler and feature configuration.
          </div>
        </div>

        {/* Result Section */}
        {result ? (
          <div className={`card ${isSpoofed ? 'result-card-spoofed' : 'result-card-normal'}`}>
            <div style={{ textAlign: 'center', marginBottom: 24 }}>
              {isSpoofed
                ? <ShieldAlert size={52} color="var(--accent-red)" style={{ margin: '0 auto 8px', display: 'block' }} />
                : <ShieldCheck size={52} color="var(--accent-green)" style={{ margin: '0 auto 8px', display: 'block' }} />
              }
              <div style={{ fontSize: '2rem', fontWeight: 800, color: isSpoofed ? 'var(--accent-red)' : 'var(--accent-green)' }}>
                {result.overall_prediction}
              </div>
              <div style={{ fontSize: '1.1rem', color: 'var(--text-secondary)', marginTop: 4 }}>
                Confidence: <strong style={{ color: 'var(--text-primary)' }}>{result.confidence?.toFixed(1)}%</strong>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 20 }}>
              {[
                { label: 'Total Sequences', value: result.total_sequences?.toLocaleString(), color: 'var(--text-primary)' },
                { label: 'Spoofed', value: result.spoofed_count?.toLocaleString(), color: 'var(--accent-red)' },
                { label: 'Normal', value: result.normal_count?.toLocaleString(), color: 'var(--accent-green)' },
                { label: 'Spoofing Rate', value: `${result.spoofing_percentage?.toFixed(1)}%`, color: 'var(--accent-yellow)' },
              ].map(({ label, value, color }) => (
                <div key={label} style={{ background: 'var(--bg-primary)', borderRadius: 8, padding: '12px 16px', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
                  <div style={{ fontSize: '1.4rem', fontWeight: 700, color }}>{value}</div>
                </div>
              ))}
            </div>

            {/* Spoofing rate bar */}
            <div style={{ marginBottom: 8, fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Spoofing Rate: {result.spoofing_percentage?.toFixed(1)}%
            </div>
            <div className="progress-bar-container" style={{ marginBottom: 16 }}>
              <div className="progress-bar-fill" style={{
                width: `${result.spoofing_percentage || 0}%`,
                background: isSpoofed ? 'linear-gradient(90deg, #ef4444, #f97316)' : 'linear-gradient(90deg, #10b981, #06b6d4)',
              }} />
            </div>
          </div>
        ) : (
          <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', minHeight: 320 }}>
            <Radar size={48} style={{ opacity: 0.2, marginBottom: 16 }} />
            <p>Detection results will appear here after running analysis.</p>
          </div>
        )}
      </div>

      {/* Confidence Distribution Chart */}
      {confidenceHistData.length > 0 && (
        <div className="chart-card" style={{ marginTop: 24 }}>
          <h3><BarChart2 size={16} /> Prediction Confidence Distribution</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={confidenceHistData}>
              <XAxis dataKey="range" stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
              <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)' }} />
              <Bar dataKey="count" name="Sequences" radius={[4, 4, 0, 0]}>
                {confidenceHistData.map((_, i) => (
                  <Cell key={i} fill={i >= 8 ? 'var(--accent-red)' : 'var(--accent-blue)'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 8 }}>
            High confidence (80-100%) sequences are highlighted in red if classified as spoofed.
          </p>
        </div>
      )}
    </div>
  )
}
