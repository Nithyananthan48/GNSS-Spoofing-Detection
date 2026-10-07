import React from 'react'
import { Satellite, Brain, Zap, Shield, BookOpen, Code2, Globe } from 'lucide-react'

const sections = [
  {
    icon: <Globe size={22} />,
    title: 'GNSS Technology',
    color: 'blue',
    content: `Global Navigation Satellite System (GNSS) refers to a constellation of satellites providing signals from space that transmit positioning and timing data to GNSS receivers. The receivers use this data to determine their location. Examples include GPS (USA), GLONASS (Russia), Galileo (EU), and BeiDou (China).

GNSS is critical infrastructure used in: autonomous vehicles, drones, aviation, maritime navigation, financial systems, smart agriculture, emergency services, military operations, and mobile networks.`,
  },
  {
    icon: <Shield size={22} />,
    title: 'GNSS Spoofing Attacks',
    color: 'red',
    content: `GNSS spoofing is a cyberattack in which an adversary transmits counterfeit GNSS signals that are stronger than authentic satellite signals, deceiving GNSS receivers into computing incorrect position, velocity, or time (PVT) solutions.

Attack consequences include:
• Autonomous vehicles being guided to wrong locations
• Drone hijacking and geofencing bypass
• Aircraft navigation errors
• Ship diversion (maritime spoofing incidents documented)
• Financial transaction timestamp manipulation
• Critical infrastructure timing disruption

Notable real-world incidents include the 2011 Iran GPS spoofing of a US RQ-170 Sentinel drone and systematic ship spoofing in the Black Sea (2017).`,
  },
  {
    icon: <Brain size={22} />,
    title: 'Deep Learning for GNSS Defense',
    color: 'purple',
    content: `Traditional spoofing detection methods (e.g., signal strength monitoring, multi-receiver comparison, AGC monitoring) are often limited to specific attack vectors and cannot generalize to sophisticated attacks.

Deep learning approaches can learn complex, high-dimensional patterns in GNSS signal data that are difficult to capture with rule-based or classical ML methods. The temporal nature of GNSS signals makes sequence models particularly effective.

Our system uses supervised classification: given a sequence of GNSS observations, the model predicts whether the observations are NORMAL or SPOOFED with associated confidence scores.`,
  },
  {
    icon: <Zap size={22} />,
    title: 'GRU and Bidirectional GRU',
    color: 'cyan',
    content: `Gated Recurrent Units (GRU) are a type of recurrent neural network (RNN) that solve the vanishing gradient problem through gating mechanisms:

• Reset Gate: controls how much past information to forget
• Update Gate: controls how much new information to incorporate

A Bidirectional GRU (Bi-GRU) processes sequences in BOTH forward and backward directions, enabling the model to capture both past AND future temporal dependencies in the GNSS signal data. This is particularly valuable because spoofing anomalies can cause correlated changes across multiple time steps.

Architecture used:
  Input(seq_len, features) → BiGRU(64) → Dropout(0.2)
  → BiGRU(32) → Dropout(0.2) → Dense(32, ReLU) → Output(Sigmoid)`,
  },
  {
    icon: <Code2 size={22} />,
    title: 'Proposed System Architecture',
    color: 'green',
    content: `The proposed system consists of:

1. Data Ingestion Layer: CSV upload with validation and format checking
2. Preprocessing Pipeline: cleaning, normalization (fitted on training data only to prevent data leakage), feature engineering
3. Feature Engineering: temporal differences, rolling statistics, position change magnitude, pseudorange rate, C/N0 anomaly
4. Sequence Generator: converts observations into overlapping time windows (length=20)
5. Bi-GRU Classifier: bidirectional GRU neural network (TensorFlow/Keras)
6. Evaluation Module: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
7. Comparison Engine: Logistic Regression, Random Forest, SVM evaluated on the same test data
8. REST API: FastAPI backend with JWT authentication
9. Dashboard: React.js cybersecurity monitoring interface`,
  },
  {
    icon: <BookOpen size={22} />,
    title: 'Research Contribution',
    color: 'yellow',
    content: `This project contributes:

• A temporal deep learning framework for GNSS spoofing detection using Bi-GRU
• A sequence-based classification approach that captures temporal dependencies in GNSS signals
• Feature engineering specifically designed for GNSS characteristics (pseudorange rates, C/N0 anomalies, position change magnitude)
• Rigorous experimental comparison with conventional ML techniques (LR, RF, SVM)
• Confidence-based spoofing identification for interpretable results
• A web-based cybersecurity monitoring dashboard suitable for operational use
• A modular architecture ready for integration with real-time GNSS receivers

Future Work:
• Real-time GNSS receiver integration (u-blox, NovAtel, Trimble)
• Multi-antenna spoofing detection
• Detection of multiple attack types (jamming, meaconing, replay)
• Federated learning for privacy-preserving distributed detection
• Edge deployment on Raspberry Pi/NVIDIA Jetson for embedded systems`,
  },
]

const colorMap = {
  blue: 'rgba(59,130,246,0.15)',
  red: 'rgba(239,68,68,0.15)',
  purple: 'rgba(139,92,246,0.15)',
  cyan: 'rgba(6,182,212,0.15)',
  green: 'rgba(16,185,129,0.15)',
  yellow: 'rgba(245,158,11,0.15)',
}
const textColorMap = {
  blue: 'var(--accent-blue)',
  red: 'var(--accent-red)',
  purple: 'var(--accent-purple)',
  cyan: 'var(--accent-cyan)',
  green: 'var(--accent-green)',
  yellow: 'var(--accent-yellow)',
}

export default function AboutPage() {
  return (
    <div>
      <div className="page-header">
        <h2><Satellite size={22} style={{ verticalAlign: 'middle', marginRight: 8 }} />About This Project</h2>
        <p>Deep Learning-Based Detection of GNSS Spoofing Attacks Using a Bidirectional GRU Network · MCA Final Year Project</p>
      </div>

      {/* Header Banner */}
      <div style={{ background: 'linear-gradient(135deg, rgba(59,130,246,0.1), rgba(6,182,212,0.05))', border: '1px solid var(--border-accent)', borderRadius: 12, padding: 32, marginBottom: 28 }}>
        <div style={{ display: 'flex', gap: 20, alignItems: 'flex-start' }}>
          <div style={{ background: 'rgba(6,182,212,0.15)', padding: 16, borderRadius: 12, flexShrink: 0 }}>
            <Satellite size={40} color="var(--accent-cyan)" />
          </div>
          <div>
            <h3 style={{ fontSize: '1.3rem', fontWeight: 700, marginBottom: 8 }}>
              Deep Learning-Based Detection of GNSS Spoofing Attacks Using a Bidirectional Gated Recurrent Unit (Bi-GRU) Network
            </h3>
            <p style={{ color: 'var(--text-secondary)', lineHeight: 1.7, fontSize: '0.9rem' }}>
              This project proposes a deep learning-based approach to enhance cybersecurity against GNSS spoofing attacks using a Bidirectional GRU neural network. The proposed model analyzes time-series GNSS signal data by learning both past and future temporal dependencies, enabling accurate identification of normal and spoofed signals.
            </p>
            <div style={{ marginTop: 16, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              {['Bi-GRU', 'TensorFlow/Keras', 'FastAPI', 'React.js', 'SQLite', 'MCA Project 2024'].map(tag => (
                <span key={tag} className="badge badge-blue">{tag}</span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Content Sections */}
      <div style={{ display: 'grid', gap: 20 }}>
        {sections.map(({ icon, title, color, content }) => (
          <div key={title} className="card">
            <div style={{ display: 'flex', gap: 14, marginBottom: 16 }}>
              <div style={{ background: colorMap[color], color: textColorMap[color], padding: 10, borderRadius: 8, flexShrink: 0, height: 'fit-content' }}>
                {icon}
              </div>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, paddingTop: 8, color: textColorMap[color] }}>{title}</h3>
            </div>
            <div style={{ color: 'var(--text-secondary)', lineHeight: 1.8, whiteSpace: 'pre-line', fontSize: '0.875rem', paddingLeft: 52 }}>
              {content}
            </div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 24, textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.82rem', padding: 20, borderTop: '1px solid var(--border-color)' }}>
        MCA Final Year Project · Department of Computer Applications · 2024–2025
      </div>
    </div>
  )
}
