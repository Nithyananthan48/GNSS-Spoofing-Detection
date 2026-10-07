import React, { useState, useEffect } from 'react'
import { predictionAPI } from '../services/api'
import { History, ShieldAlert, ShieldCheck, RefreshCw } from 'lucide-react'

export default function PredictionHistoryPage() {
  const [predictions, setPredictions] = useState([])
  const [loading, setLoading] = useState(true)

  const load = () => {
    setLoading(true)
    predictionAPI.history()
      .then(r => setPredictions(r.data))
      .catch(() => setPredictions([]))
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h2><History size={22} style={{ verticalAlign: 'middle', marginRight: 8 }} />Prediction History</h2>
          <p>All GNSS spoofing detection sessions run under your account.</p>
        </div>
        <button className="btn btn-outline btn-sm" onClick={load} disabled={loading}>
          <RefreshCw size={14} />  Refresh
        </button>
      </div>

      {loading ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: 60 }}>
          <div className="loading-spinner" style={{ width: 36, height: 36, borderWidth: 3 }} />
        </div>
      ) : predictions.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: 60 }}>
          <History size={48} style={{ margin: '0 auto 16px', display: 'block', opacity: 0.25, color: 'var(--text-muted)' }} />
          <h3 style={{ color: 'var(--text-secondary)', marginBottom: 8 }}>No Predictions Yet</h3>
          <p style={{ color: 'var(--text-muted)' }}>Run a detection on the Detection page to see results here.</p>
        </div>
      ) : (
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>File</th>
                <th>Result</th>
                <th>Confidence</th>
                <th>Normal</th>
                <th>Spoofed</th>
                <th>Spoofing %</th>
                <th>Total Sequences</th>
                <th>Date</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((p, idx) => (
                <tr key={p.id}>
                  <td style={{ color: 'var(--text-muted)' }}>{idx + 1}</td>
                  <td>
                    <div style={{ fontWeight: 600 }}>{p.filename}</div>
                  </td>
                  <td>
                    <span className={`badge ${p.prediction === 'SPOOFED' ? 'badge-spoofed' : 'badge-normal'}`}>
                      {p.prediction === 'SPOOFED'
                        ? <><ShieldAlert size={12} /> SPOOFED</>
                        : <><ShieldCheck size={12} /> NORMAL</>}
                    </span>
                  </td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{
                        width: 48,
                        height: 6,
                        background: 'var(--bg-primary)',
                        borderRadius: 3,
                        overflow: 'hidden',
                        border: '1px solid var(--border-color)',
                      }}>
                        <div style={{
                          width: `${p.confidence}%`,
                          height: '100%',
                          background: p.prediction === 'SPOOFED' ? 'var(--accent-red)' : 'var(--accent-green)',
                          borderRadius: 3,
                        }} />
                      </div>
                      <span style={{ fontSize: '0.875rem', fontWeight: 600 }}>{p.confidence?.toFixed(1)}%</span>
                    </div>
                  </td>
                  <td style={{ color: 'var(--accent-green)' }}>{p.normal_count?.toLocaleString()}</td>
                  <td style={{ color: 'var(--accent-red)' }}>{p.spoofed_count?.toLocaleString()}</td>
                  <td>
                    <span style={{ color: p.spoofing_percentage > 20 ? 'var(--accent-red)' : 'var(--accent-green)', fontWeight: 700 }}>
                      {p.spoofing_percentage?.toFixed(1)}%
                    </span>
                  </td>
                  <td>{p.total_count?.toLocaleString()}</td>
                  <td style={{ color: 'var(--text-muted)', fontSize: '0.8rem', whiteSpace: 'nowrap' }}>
                    {p.created_at ? new Date(p.created_at).toLocaleString() : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {predictions.length > 0 && (
        <div style={{ marginTop: 12, color: 'var(--text-muted)', fontSize: '0.82rem' }}>
          Showing last {predictions.length} detection{predictions.length !== 1 ? 's' : ''}.
        </div>
      )}
    </div>
  )
}
