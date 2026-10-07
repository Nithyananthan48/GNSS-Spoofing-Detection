# MCA Final Year Project Documentation
## Deep Learning-Based Detection of GNSS Spoofing Attacks Using a Bidirectional Gated Recurrent Unit (Bi-GRU) Network

---

## 1. Introduction

Global Navigation Satellite System (GNSS) constellations (such as GPS, GLONASS, Galileo, and BeiDou) form the positioning, navigation, and timing (PNT) backbone for modern autonomous vehicles, commercial aviation, maritime shipping, financial transaction timestamping, telecommunications, and defense infrastructure. 

However, civil GNSS signals are unencrypted and transmit at low power levels (~-160 dBW at Earth's surface). This fundamental physical vulnerability enables **spoofing attacks**: malicious transmission of counterfeit RF signals that overpower authentic satellite broadcasts and deceive receivers into computing fabricated position, velocity, and timing (PVT) solutions.

---

## 2. Problem Statement & Research Motivation

Traditional anti-spoofing techniques rely primarily on:
- Receiver Autonomous Integrity Monitoring (RAIM)
- Antenna array angle-of-arrival (AoA) estimation
- Automatic Gain Control (AGC) power anomaly detection

**Limitations of Traditional Approaches:**
1. Susceptible to sophisticated power-matched spoofing where adversary signal levels match authentic signals.
2. Inability to capture multi-observation temporal dynamics across time-series windows.
3. High false alarm rates in dense urban multipath environments.
4. Classical ML models (such as Logistic Regression or SVM) treat observations as independent points, discarding sequential correlations.

**Proposed Solution:**
A temporal deep learning approach leveraging a **Bidirectional Gated Recurrent Unit (Bi-GRU)** neural network that learns both past and future sequential dependencies across time-series windows of GNSS observables.

---

## 3. Mathematical Formulation & Architecture

### 3.1 Feature Vector & Observable Matrix
At each epoch $t$, a GNSS observation vector $x_t \in \mathbb{R}^D$ is collected containing raw observables and engineered temporal differences:

$$
x_t = \left[ \rho_t, \Phi_t, \text{C/N0}_t, f_{D,t}, \theta_{el,t}, \theta_{az,t}, \phi_t, \lambda_t, h_t, v_t, \Delta \rho_t, \Delta \text{C/N0}_t, \dots \right]
$$

Over a sliding window of length $L = 20$:
$$
X_i = [x_{i}, x_{i+1}, \dots, x_{i+L-1}] \in \mathbb{R}^{L \times D}
$$

### 3.2 Bidirectional Gated Recurrent Unit (Bi-GRU)
For each time step $t \in [1, L]$, the forward GRU $\overrightarrow{h}_t$ and backward GRU $\overleftarrow{h}_t$ compute hidden representations:

$$
\begin{aligned}
r_t &= \sigma(W_r x_t + U_r h_{t-1} + b_r) \quad &\text{(Reset Gate)} \\
z_t &= \sigma(W_z x_t + U_z h_{t-1} + b_z) \quad &\text{(Update Gate)} \\
\tilde{h}_t &= \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h) \quad &\text{(Candidate State)} \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad &\text{(Hidden State)}
\end{aligned}
$$

The bidirectional output concatenates forward and backward states:
$$
h_t^{\text{bi}} = [\overrightarrow{h}_t \,\|\, \overleftarrow{h}_t]
$$

### 3.3 Deep Neural Network Topology
```
Input: Tensor of shape (BatchSize, 20, 31)
  │
  ▼
Bidirectional GRU Layer 1 (64 units per direction, output dim = 128)
  │
  ▼
Dropout (p = 0.2)
  │
  ▼
Bidirectional GRU Layer 2 (32 units per direction, output dim = 64)
  │
  ▼
Dropout (p = 0.2)
  │
  ▼
Dense / Linear Layer (32 units, ReLU activation)
  │
  ▼
Output Layer (1 unit, Sigmoid activation) -> P(Spoofed | X)
```

---

## 4. Preprocessing & Feature Engineering

To prevent **data leakage**, the normalization pipeline adheres to strict machine learning discipline:
1. Data split into Train (65%), Validation (20%), and Test (15%) sets using reproducible random seeds.
2. `StandardScaler` calculates $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ **strictly from the training partition**.
3. Validation and test partitions are transformed using the precomputed parameters.
4. Fitted scaler serialized to `trained_models/scaler.pkl` for production inference.

### Engineered Features:
- **First-order temporal differences:** $\Delta \rho = \rho_t - \rho_{t-1}$, $\Delta \text{C/N0}$, $\Delta f_D$
- **Rolling statistical moments:** Rolling mean and standard deviation ($W=5$)
- **Position change magnitude:** $\Delta s = \sqrt{(\Delta \phi \cdot R)^2 + (\Delta \lambda \cdot R \cos \phi)^2 + \Delta h^2}$
- **Pseudorange Rate:** $\dot{\rho} = \Delta \rho / \Delta t$
- **C/N0 Anomaly:** Deviation from 10-epoch moving average

---

## 5. Experimental Results & Model Comparison

All models were evaluated on the **identical test set** (731 temporal sequences, 5,000 raw samples):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 99.45% | 98.92% | 100.00% | 0.9932 | 0.9999 |
| **Random Forest** | 99.32% | 98.41% | 100.00% | 0.9915 | 1.0000 |
| **Support Vector Machine (SVM)** | 93.71% | 85.00% | 100.00% | 0.9184 | 0.9906 |
| **Bi-GRU (Proposed)** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **1.0000** |

### Key Observations:
- **Bi-GRU Superiority:** By capturing bidirectional temporal sequence dependencies, the Bi-GRU model effectively eliminates false positives and false negatives under simulated spoofing anomalies.
- **Fast Response Time:** Model inference on a 5,000-sample dataset takes under 1.2 seconds on standard CPU hardware.

---

## 6. System Architecture & Component Interactions

```
┌─────────────────────────────────────────────────────────────────┐
│                       React 18 Dashboard                        │
│  (Overview, Upload, Model Training, Detection, Evaluation)      │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTP / JSON / Multipart
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FastAPI Backend                           │
│  ├── JWT Authentication & User Management                       │
│  ├── Dataset Upload & Validation Engine                         │
│  ├── Background Training Task Manager                           │
│  └── Real-time Prediction & Inference Service                   │
└──────────────┬──────────────────────────────────┬───────────────┘
               │                                  │
               ▼                                  ▼
┌──────────────────────────────┐   ┌──────────────────────────────┐
│       SQLite Database        │   │    PyTorch Bi-GRU Engine     │
│  - Users                     │   │  - bigru_model.pt            │
│  - Datasets                  │   │  - scaler.pkl                │
│  - Training Runs             │   │  - model_metadata.json       │
│  - Predictions History       │   │  - feature_config.json       │
└──────────────────────────────┘   └──────────────────────────────┘
```

---

## 7. Viva / Presentation Q&A Guide

**Q1: Why choose Bi-GRU over standard LSTM or unidirectional GRU?**  
*Answer:* GRUs have fewer parameters than LSTMs (reset and update gates vs. input, forget, and output gates), leading to faster training and lower inference latency suitable for edge-deployed receivers. Bidirectionality enables the network to analyze context both leading up to an anomaly and immediately following it.

**Q2: How does the system prevent data leakage?**  
*Answer:* Preprocessing scales features using `StandardScaler` fitted solely on the training split. Sequence generation and test evaluations are strictly partitioned.

**Q3: How does the model perform in real-time?**  
*Answer:* The inference service takes sliding windows of 20 observations and executes batch matrix multiplication in PyTorch in under 2ms per window, enabling real-time detection on 1Hz to 10Hz GNSS data streams.

---

## 8. Limitations & Future Work

1. **Hardware In-the-Loop Testing:** Direct RF front-end integration with SDRs (Software Defined Radios like HackRF, USRP) and GNSS receivers (u-blox ZED-F9P).
2. **Multi-constellation Multi-frequency (MCMF):** Expanding feature sets to dual-frequency L1/L2 and L1/L5 carrier observables.
3. **Adversarial Spoofing Attacks:** Evaluation against adaptive adversaries capable of gradually drifting carrier phases.
