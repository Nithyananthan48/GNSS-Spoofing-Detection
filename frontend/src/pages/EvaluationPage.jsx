import React, { useState, useEffect } from 'react'
import { modelAPI } from '../services/api'
import { BarChart2, Award } from 'lucide-react'
import {
  RadarChart, PolarGrid, PolarAngleAxis, Radar as RadarShape, ResponsiveContainer, Tooltip,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend, Cell
} from 'recharts'

export default function EvaluationPage() {
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    modelAPI.metrics()
      .then(r => setMetrics(r.data))
      .catch(() => setMetrics({ trained: false }))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ display: 'flex', justifyContent: 'center', padding: 80 }}><div className="loading-spinner" style={{ width: 40, height: 40, borderWidth: 3 }} /></div>

  if (!metrics?.trained) {
    return (
      <div>
        <div className="page-header">
          <h2><BarChart2 size={22} style={{ verticalAlign: 'middle', marginRight: 8 }} />Model Evaluation</h2>
        </div>
        <div className="card" style={{ textAlign: 'center', padding: 60 }}>
          <BarChart2 size={48} style={{ margin: '0 auto 16px', display: 'block', opacity: 0.3, color: 'var(--text-muted)' }} />
          <h3 style={{ color: 'var(--text-secondary)', marginBottom: 8 }}>Model Not Trained – Metrics Unavailable</h3>
          <p style={{ color: 'var(--text-muted)' }}>Please upload a dataset and train the Bi-GRU model first.</p>
        </div>
      </div>
    )
  }

  const bigru = metrics.bigru_metrics || {}
  const baseline = metrics.baseline_metrics || {}

  // Confusion matrix
  const cm = bigru.confusion_matrix || [[0, 0], [0, 0]]
  const tn = cm[0]?.[0] || 0
  const fp = cm[0]?.[1] || 0
  const fn = cm[1]?.[0] || 0
  const tp = cm[1]?.[1] || 0

  // Comparison table data
  const comparisonData = [
    ...Object.entries(baseline).filter(([, v]) => !v.error).map(([name, m]) => ({
      name, accuracy: +(m.accuracy * 100).toFixed(2), precision: +(m.precision * 100).toFixed(2),
      recall: +(m.recall * 100).toFixed(2), f1: +(m.f1_score * 100).toFixed(2), roc: +(m.roc_auc * 100).toFixed(2),
    })),
    {
      name: 'Bi-GRU (Proposed)',
      accuracy: +(bigru.accuracy * 100).toFixed(2),
      precision: +(bigru.precision * 100).toFixed(2),
      recall: +(bigru.recall * 100).toFixed(2),
      f1: +(bigru.f1_score * 100).toFixed(2),
      roc: +(bigru.roc_auc * 100).toFixed(2),
    },
  ]

  // Radar chart data
  const radarData = [
    { metric: 'Accuracy', value: +(bigru.accuracy * 100).toFixed(1) },
    { metric: 'Precision', value: +(bigru.precision * 100).toFixed(1) },
    { metric: 'Recall', value: +(bigru.recall * 100).toFixed(1) },
    { metric: 'F1 Score', value: +(bigru.f1_score * 100).toFixed(1) },
    { metric: 'ROC-AUC', value: +(bigru.roc_auc * 100).toFixed(1) },
  ]

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#06b6d4']

  return (
    <div>
      <div className="page-header">
        <h2><BarChart2 size={22} style={{ verticalAlign: 'middle', marginRight: 8 }} />Model Evaluation</h2>
        <p>Bi-GRU performance metrics and comparison with baseline ML models on the same test dataset.</p>
      </div>

      {/* Bi-GRU Metrics */}
      <div className="metrics-grid" style={{ marginBottom: 28 }}>
        {[
          { label: 'Accuracy', value: `${(bigru.accuracy * 100).toFixed(2)}%` },
          { label: 'Precision', value: `${(bigru.precision * 100).toFixed(2)}%` },
          { label: 'Recall', value: `${(bigru.recall * 100).toFixed(2)}%` },
          { label: 'F1 Score', value: `${(bigru.f1_score * 100).toFixed(2)}%` },
          { label: 'ROC-AUC', value: bigru.roc_auc?.toFixed(4) },
          { label: 'False Positive Rate', value: `${(bigru.false_positive_rate * 100).toFixed(2)}%` },
          { label: 'False Negative Rate', value: `${(bigru.false_negative_rate * 100).toFixed(2)}%` },
          { label: 'Epochs Trained', value: metrics.epochs_trained },
        ].map(m => (
          <div key={m.label} style={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)', borderRadius: 10, padding: '16px 20px' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>{m.label}</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{m.value}</div>
          </div>
        ))}
      </div>

      <div className="charts-grid">
        {/* Confusion Matrix */}
        <div className="chart-card">
          <h3><Award size={16} /> Confusion Matrix</h3>
          <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start' }}>
            <div className="confusion-matrix">
              <div className="cm-label"></div>
              <div className="cm-label">Pred Normal</div>
              <div className="cm-label">Pred Spoofed</div>
              <div className="cm-label" style={{ writingMode: 'vertical-lr', transform: 'rotate(180deg)', padding: '8px 4px' }}>Actual Normal</div>
              <div className="cm-cell tn" title="True Negative">{tn.toLocaleString()}<div style={{ fontSize: '0.65rem', marginTop: 4, opacity: 0.7 }}>TN</div></div>
              <div className="cm-cell fp" title="False Positive">{fp.toLocaleString()}<div style={{ fontSize: '0.65rem', marginTop: 4, opacity: 0.7 }}>FP</div></div>
              <div className="cm-label" style={{ writingMode: 'vertical-lr', transform: 'rotate(180deg)', padding: '8px 4px' }}>Actual Spoofed</div>
              <div className="cm-cell fn" title="False Negative">{fn.toLocaleString()}<div style={{ fontSize: '0.65rem', marginTop: 4, opacity: 0.7 }}>FN</div></div>
              <div className="cm-cell tp" title="True Positive">{tp.toLocaleString()}<div style={{ fontSize: '0.65rem', marginTop: 4, opacity: 0.7 }}>TP</div></div>
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 2 }}>
              <div><span style={{ color: 'var(--accent-green)' }}>TN</span> True Negative (Normal → Normal)</div>
              <div><span style={{ color: 'var(--accent-red)' }}>FP</span> False Positive (Normal → Spoofed)</div>
              <div><span style={{ color: 'var(--accent-yellow)' }}>FN</span> False Negative (Spoofed → Normal)</div>
              <div><span style={{ color: 'var(--accent-green)' }}>TP</span> True Positive (Spoofed → Spoofed)</div>
            </div>
          </div>
        </div>

        {/* Radar Chart */}
        <div className="chart-card">
          <h3>Bi-GRU Performance Radar</h3>
          <ResponsiveContainer width="100%" height={260}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="var(--border-color)" />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12, fill: 'var(--text-secondary)' }} />
              <RadarShape name="Bi-GRU" dataKey="value" stroke="var(--accent-blue)" fill="var(--accent-blue)" fillOpacity={0.25} />
              <Tooltip formatter={v => `${v}%`} contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)' }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model Comparison Bar Chart */}
      {comparisonData.length > 1 && (
        <div className="chart-card" style={{ marginTop: 20 }}>
          <h3>Model Comparison – Accuracy vs F1 Score</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={comparisonData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
              <XAxis dataKey="name" stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
              <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} unit="%" domain={[0, 105]} />
              <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)' }} formatter={v => `${v}%`} />
              <Legend />
              <Bar dataKey="accuracy" name="Accuracy" radius={[4, 4, 0, 0]}>
                {comparisonData.map((d, i) => <Cell key={i} fill={d.name.includes('Bi-GRU') ? 'var(--accent-cyan)' : COLORS[i % COLORS.length]} />)}
              </Bar>
              <Bar dataKey="f1" name="F1 Score" radius={[4, 4, 0, 0]}>
                {comparisonData.map((d, i) => <Cell key={i} fill={d.name.includes('Bi-GRU') ? 'var(--accent-green)' : '#64748b'} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Comparison Table */}
      <div className="card" style={{ marginTop: 20 }}>
        <div className="card-title">Model Comparison Table</div>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
                <th>ROC-AUC</th>
              </tr>
            </thead>
            <tbody>
              {comparisonData.map(m => (
                <tr key={m.name}>
                  <td>
                    <span style={{ fontWeight: m.name.includes('Bi-GRU') ? 700 : 400, color: m.name.includes('Bi-GRU') ? 'var(--accent-cyan)' : 'inherit' }}>
                      {m.name.includes('Bi-GRU') ? '⭐ ' : ''}{m.name}
                    </span>
                  </td>
                  <td>{m.accuracy}%</td>
                  <td>{m.precision}%</td>
                  <td>{m.recall}%</td>
                  <td>{m.f1}%</td>
                  <td>{(m.roc / 100).toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ marginTop: 12, fontSize: '0.78rem', color: 'var(--text-muted)' }}>
          ⭐ Proposed model. All models evaluated on the same test dataset split.
        </div>
      </div>
    </div>
  )
}
