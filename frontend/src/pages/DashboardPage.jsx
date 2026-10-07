import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { dashboardAPI } from '../services/api'
import {
  Database, ShieldAlert, ShieldCheck, BarChart3,
  Cpu, Upload, Radar, TrendingUp, Clock
} from 'lucide-react'
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

export default function DashboardPage() {
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    dashboardAPI.summary()
      .then(res => setSummary(res.data))
      .catch(err => setError(err.response?.data?.detail || 'Failed to load dashboard data.'))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div style={{ display: 'flex', justifyContent: 'center', padding: 80 }}>
      <div className="loading-spinner" style={{ width: 40, height: 40, borderWidth: 3 }} />
    </div>
  )

  const s = summary || {}
  const pieData = [
    { name: 'Normal', value: s.total_normal_detected || 0, color: '#10b981' },
    { name: 'Spoofed', value: s.total_spoofed_detected || 0, color: '#ef4444' },
  ]

  return (
    <div>
      <div className="page-header">
        <h2>🛰️ System Dashboard</h2>
        <p>Deep Learning-Based GNSS Spoofing Detection · Bi-GRU Neural Network</p>
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      {/* Status Banner */}
      <div className={`alert ${s.model_trained ? 'alert-success' : 'alert-warning'}`} style={{ marginBottom: 24 }}>
        <strong>{s.model_trained ? '✅ Model Ready' : '⚠️ Model Not Trained'}</strong>
        {s.model_trained
          ? ` – Bi-GRU model is trained and ready. Detection accuracy: ${s.model_accuracy}%`
          : ' – Please upload a dataset and train the model to enable detection.'}
      </div>

      {/* Metric Cards */}
      <div className="metrics-grid">
        <MetricCard icon={<Database size={22} />} iconClass="blue" label="Datasets Uploaded" value={s.total_datasets || 0} sub="CSV files" />
        <MetricCard icon={<Upload size={22} />} iconClass="cyan" label="Total GNSS Samples" value={(s.total_samples_uploaded || 0).toLocaleString()} sub="observations" />
        <MetricCard icon={<ShieldCheck size={22} />} iconClass="green" label="Normal Detected" value={(s.total_normal_detected || 0).toLocaleString()} sub="clean signals" />
        <MetricCard icon={<ShieldAlert size={22} />} iconClass="red" label="Spoofed Detected" value={(s.total_spoofed_detected || 0).toLocaleString()} sub="attack signals" />
        <MetricCard icon={<Radar size={22} />} iconClass="purple" label="Predictions Run" value={s.total_predictions_run || 0} sub="detection sessions" />
        <MetricCard
          icon={<TrendingUp size={22} />}
          iconClass="yellow"
          label="Model Accuracy"
          value={s.model_trained ? `${s.model_accuracy}%` : '—'}
          sub={s.model_trained ? 'Bi-GRU test accuracy' : 'Train model first'}
        />
      </div>

      <div className="charts-grid">
        {/* Detection Distribution */}
        <div className="chart-card">
          <h3><ShieldAlert size={16} /> Detection Distribution</h3>
          {(s.total_normal_detected || 0) + (s.total_spoofed_detected || 0) > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={pieData} cx="50%" cy="50%" outerRadius={90} dataKey="value" label={({ name, percent }) => `${name} ${(percent * 100).toFixed(1)}%`}>
                  {pieData.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                </Pie>
                <Tooltip formatter={(v) => v.toLocaleString()} contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)' }} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ textAlign: 'center', padding: 60, color: 'var(--text-muted)' }}>
              <Radar size={40} style={{ margin: '0 auto 12px', display: 'block', opacity: 0.3 }} />
              <p>No detections yet. Run a detection to see results.</p>
            </div>
          )}
        </div>

        {/* Recent Predictions */}
        <div className="chart-card">
          <h3><Clock size={16} /> Recent Predictions</h3>
          {s.recent_predictions?.length > 0 ? (
            <div style={{ overflow: 'hidden' }}>
              {s.recent_predictions.map(pred => (
                <div key={pred.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 0', borderBottom: '1px solid var(--border-color)' }}>
                  <div>
                    <div style={{ fontSize: '0.875rem', fontWeight: 600 }}>{pred.filename}</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      {pred.spoofing_percentage?.toFixed(1)}% spoofed · {new Date(pred.created_at).toLocaleString()}
                    </div>
                  </div>
                  <span className={`badge ${pred.prediction === 'SPOOFED' ? 'badge-spoofed' : 'badge-normal'}`}>
                    {pred.prediction}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: 60, color: 'var(--text-muted)' }}>
              <Clock size={40} style={{ margin: '0 auto 12px', display: 'block', opacity: 0.3 }} />
              <p>No predictions yet.</p>
            </div>
          )}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card" style={{ marginTop: 8 }}>
        <div className="card-title">Quick Actions</div>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          <Link to="/upload" className="btn btn-outline"><Upload size={16} /> Upload Dataset</Link>
          <Link to="/train" className="btn btn-outline"><Cpu size={16} /> Train Model</Link>
          <Link to="/detect" className="btn btn-primary"><Radar size={16} /> Run Detection</Link>
          <Link to="/evaluate" className="btn btn-outline"><BarChart3 size={16} /> View Metrics</Link>
        </div>
      </div>
    </div>
  )
}

function MetricCard({ icon, iconClass, label, value, sub }) {
  return (
    <div className="metric-card">
      <div className={`metric-icon ${iconClass}`}>{icon}</div>
      <div className="metric-info">
        <div className="metric-label">{label}</div>
        <div className="metric-value">{value}</div>
        {sub && <div className="metric-sub">{sub}</div>}
      </div>
    </div>
  )
}
