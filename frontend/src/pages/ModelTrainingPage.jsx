import React, { useState, useEffect, useRef } from 'react'
import { datasetAPI, modelAPI } from '../services/api'
import { Cpu, Play, RefreshCw, CheckCircle, XCircle, Clock } from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'

const POLL_INTERVAL = 3000

export default function ModelTrainingPage() {
  const [datasets, setDatasets] = useState([])
  const [selectedDataset, setSelectedDataset] = useState('')
  const [status, setStatus] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [training, setTraining] = useState(false)
  const [error, setError] = useState('')
  const [info, setInfo] = useState('')
  const pollRef = useRef(null)

  useEffect(() => {
    datasetAPI.list().then(r => setDatasets(r.data)).catch(() => {})
    loadMetrics()
    checkStatus()
  }, [])

  const loadMetrics = () => {
    modelAPI.metrics().then(r => setMetrics(r.data)).catch(() => {})
  }

  const checkStatus = () => {
    modelAPI.status().then(r => {
      setStatus(r.data)
      if (r.data.status === 'training' || r.data.status === 'preprocessing' || r.data.status === 'evaluating' || r.data.status === 'baseline_models' || r.data.status === 'building_model' || r.data.status === 'saving') {
        setTraining(true)
      } else {
        setTraining(false)
        if (r.data.status === 'completed') loadMetrics()
      }
    }).catch(() => {})
  }

  const startPolling = () => {
    pollRef.current = setInterval(() => {
      modelAPI.status().then(r => {
        setStatus(r.data)
        const done = !['preprocessing', 'building_model', 'training', 'evaluating', 'baseline_models', 'saving'].includes(r.data.status)
        if (done) {
          clearInterval(pollRef.current)
          setTraining(false)
          if (r.data.status === 'completed') {
            loadMetrics()
            setInfo('✅ Training complete! Model metrics updated.')
          } else if (r.data.status === 'failed') {
            setError(`Training failed: ${r.data.message}`)
          }
        }
      })
    }, POLL_INTERVAL)
  }

  useEffect(() => () => clearInterval(pollRef.current), [])

  const handleTrain = async () => {
    if (!selectedDataset) { setError('Please select a dataset first.'); return }
    setError(''); setInfo('')
    setTraining(true)
    try {
      await modelAPI.train(parseInt(selectedDataset))
      setInfo('Training started in background. Monitoring progress...')
      startPolling()
    } catch (err) {
      setTraining(false)
      setError(err.response?.data?.detail || 'Failed to start training.')
    }
  }

  // Build chart data from history
  const historyData = (() => {
    if (!metrics?.training_history) return []
    const loss = metrics.training_history.loss || []
    const val_loss = metrics.training_history.val_loss || []
    const acc = metrics.training_history.accuracy || []
    const val_acc = metrics.training_history.val_accuracy || []
    return loss.map((l, i) => ({
      epoch: i + 1,
      loss: parseFloat(l.toFixed(4)),
      val_loss: parseFloat((val_loss[i] || 0).toFixed(4)),
      accuracy: parseFloat(((acc[i] || 0) * 100).toFixed(2)),
      val_accuracy: parseFloat(((val_acc[i] || 0) * 100).toFixed(2)),
    }))
  })()

  const bigru = metrics?.bigru_metrics || {}

  return (
    <div>
      <div className="page-header">
        <h2><Cpu size={22} style={{ verticalAlign: 'middle', marginRight: 8 }} />Model Training</h2>
        <p>Train the Bi-GRU neural network on your GNSS dataset. Training runs in the background.</p>
      </div>

      {/* Training Controls */}
      <div className="card mb-6">
        <div className="card-title">Start Training</div>

        {error && <div className="alert alert-error">{error}</div>}
        {info  && <div className="alert alert-info">{info}</div>}

        <div style={{ display: 'flex', gap: 16, alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div className="form-group" style={{ flex: 1, minWidth: 240, marginBottom: 0 }}>
            <label className="form-label">Select Training Dataset</label>
            <select
              className="form-input"
              value={selectedDataset}
              onChange={e => setSelectedDataset(e.target.value)}
              disabled={training}
            >
              <option value="">-- Choose a dataset --</option>
              {datasets.map(d => (
                <option key={d.id} value={d.id}>{d.filename} ({d.rows_count?.toLocaleString()} rows)</option>
              ))}
            </select>
          </div>
          <button className="btn btn-primary btn-lg" onClick={handleTrain} disabled={training || !selectedDataset}>
            {training ? <><span className="loading-spinner" /> Training...</> : <><Play size={16} /> Start Training</>}
          </button>
          <button className="btn btn-outline" onClick={checkStatus} title="Refresh status">
            <RefreshCw size={16} />
          </button>
        </div>

        {datasets.length === 0 && (
          <div style={{ marginTop: 12, color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            No datasets found. <a href="/upload" style={{ color: 'var(--accent-blue)' }}>Upload a dataset first.</a>
          </div>
        )}
      </div>

      {/* Training Status */}
      {status && (
        <div className="card mb-6">
          <div className="card-title">Training Status</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
            {status.status === 'completed' && <CheckCircle size={20} color="var(--accent-green)" />}
            {status.status === 'failed'    && <XCircle size={20} color="var(--accent-red)" />}
            {!['completed','failed','idle'].includes(status.status) && <div className="loading-spinner" />}
            <div>
              <div style={{ fontWeight: 700, textTransform: 'capitalize', color: status.status === 'failed' ? 'var(--accent-red)' : status.status === 'completed' ? 'var(--accent-green)' : 'var(--text-primary)' }}>
                {status.status?.replace('_', ' ')}
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{status.message}</div>
            </div>
          </div>
          <div className="progress-bar-container">
            <div className="progress-bar-fill" style={{ width: `${status.progress || 0}%` }} />
          </div>
          <div style={{ textAlign: 'right', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 4 }}>{status.progress || 0}%</div>
        </div>
      )}

      {/* Metrics */}
      {metrics?.trained && (
        <>
          <div className="metrics-grid" style={{ marginBottom: 24 }}>
            {[
              { label: 'Accuracy', value: `${(bigru.accuracy * 100).toFixed(2)}%`, color: 'blue' },
              { label: 'Precision', value: `${(bigru.precision * 100).toFixed(2)}%`, color: 'green' },
              { label: 'Recall', value: `${(bigru.recall * 100).toFixed(2)}%`, color: 'cyan' },
              { label: 'F1 Score', value: `${(bigru.f1_score * 100).toFixed(2)}%`, color: 'purple' },
              { label: 'ROC-AUC', value: bigru.roc_auc?.toFixed(4), color: 'yellow' },
              { label: 'Epochs Trained', value: metrics.epochs_trained, color: 'red' },
            ].map(m => (
              <div className="metric-card" key={m.label}>
                <div className={`metric-icon ${m.color}`}><BarIcon /></div>
                <div className="metric-info">
                  <div className="metric-label">{m.label}</div>
                  <div className="metric-value" style={{ fontSize: '1.4rem' }}>{m.value}</div>
                </div>
              </div>
            ))}
          </div>

          {historyData.length > 0 && (
            <div className="charts-grid">
              <div className="chart-card">
                <h3>Training vs Validation Accuracy</h3>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={historyData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                    <XAxis dataKey="epoch" stroke="var(--text-muted)" tick={{ fontSize: 11 }} label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: 'var(--text-muted)', fontSize: 11 }} />
                    <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} unit="%" domain={[0, 100]} />
                    <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)', fontSize: 12 }} formatter={v => `${v}%`} />
                    <Legend />
                    <Line type="monotone" dataKey="accuracy" name="Train Accuracy" stroke="var(--accent-blue)" dot={false} strokeWidth={2} />
                    <Line type="monotone" dataKey="val_accuracy" name="Val Accuracy" stroke="var(--accent-cyan)" dot={false} strokeWidth={2} strokeDasharray="5 3" />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="chart-card">
                <h3>Training vs Validation Loss</h3>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={historyData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                    <XAxis dataKey="epoch" stroke="var(--text-muted)" tick={{ fontSize: 11 }} label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: 'var(--text-muted)', fontSize: 11 }} />
                    <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
                    <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)', fontSize: 12 }} />
                    <Legend />
                    <Line type="monotone" dataKey="loss" name="Train Loss" stroke="var(--accent-red)" dot={false} strokeWidth={2} />
                    <Line type="monotone" dataKey="val_loss" name="Val Loss" stroke="var(--accent-yellow)" dot={false} strokeWidth={2} strokeDasharray="5 3" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </>
      )}

      {!metrics?.trained && (
        <div className="card" style={{ textAlign: 'center', padding: 60, color: 'var(--text-muted)' }}>
          <Cpu size={48} style={{ margin: '0 auto 16px', display: 'block', opacity: 0.3 }} />
          <h3 style={{ color: 'var(--text-secondary)', marginBottom: 8 }}>Model Not Trained Yet</h3>
          <p>Select a dataset and click "Start Training" to train the Bi-GRU model.</p>
        </div>
      )}
    </div>
  )
}

function BarIcon() {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="10" width="4" height="11" /><rect x="10" y="4" width="4" height="17" /><rect x="17" y="7" width="4" height="14" /></svg>
}
