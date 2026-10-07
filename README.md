# Deep Learning-Based Detection of GNSS Spoofing Attacks Using a Bidirectional Gated Recurrent Unit (Bi-GRU) Network

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-EE4C2C.svg)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF.svg)](https://vitejs.dev/)
[![License](https://img.shields.io/badge/License-Academic%20%2F%20MIT-green.svg)](#license)

---

## 📌 Project Overview

Global Navigation Satellite System (GNSS) signals—including GPS, GLONASS, Galileo, and BeiDou—are the backbone of modern positioning, navigation, and timing (PNT) infrastructures. Critical domains such as autonomous vehicles, commercial aviation, maritime shipping, telecommunications, power grids, and financial transaction networks rely entirely on GNSS timing and positioning integrity.

However, civilian GNSS broadcast signals are unencrypted, low-power (~ -160 dBW at Earth's surface), and structurally predictable. This makes them severely vulnerable to **GNSS Spoofing Attacks**, where a malicious transmitter broadcasts counterfeit satellite signals with higher power, hijacking the victim receiver's tracking loops and manipulating calculated position, velocity, and time (PVT) data.

This project implements an end-to-end cybersecurity solution powered by a **Bidirectional Gated Recurrent Unit (Bi-GRU)** deep neural network. By processing multi-parametric temporal sequences of GNSS observables, the system captures both past and future temporal context, delivering real-time spoofing detection with high accuracy and low false-alarm rates.

---

## 🎯 Problem Statement

GNSS spoofing is far more hazardous than jamming (denial-of-service):
1. **Stealth**: Receivers do not lose lock immediately; they silently follow spoofed pseudo-trajectories.
2. **Autonomous Systems Vulnerability**: Self-driving vehicles, delivery drones, and automated vessels can be diverted into danger or hijacked.
3. **Critical Infrastructure Disruption**: Power grid synchronizers and financial timestamping engines experience clock drift and data integrity failures.
4. **Failure of Traditional Threshold Methods**: Conventional rule-based signal-to-noise ratio (SNR) or Doppler discrepancy checks fail against sophisticated, coherent, and power-matched spoofers.

**Solution**: Time-series deep learning using Bidirectional GRUs that analyze sequential GNSS observables (pseudorange, carrier phase, Doppler shifts, C/N0 fluctuations) to accurately classify signals as **Normal (0)** or **Spoofed (1)**.

---

## ✨ Features

- **Bidirectional GRU Architecture**: Processes sliding windows of length 20 to capture bidirectional temporal dependencies in GNSS signal dynamics.
- **Comprehensive GNSS Observables**: Ingests pseudorange, carrier phase, carrier-to-noise ratio ($C/N_0$), Doppler frequency shift, elevation, azimuth, latitude, longitude, altitude, and velocity.
- **Interactive Web Interface**: High-performance React 18 dashboard built with Vite, Tailwind-styled components, and Recharts visualization.
- **FastAPI REST Backend**: Asynchronous, high-throughput backend with auto-generated OpenAPI/Swagger documentation.
- **Data Validation & Leakage-Free Preprocessing**: Scikit-Learn `StandardScaler` pipeline fitted strictly on training splits to prevent data leakage.
- **Comparative Baseline Benchmarks**: Automated side-by-side performance evaluation comparing Bi-GRU against Logistic Regression, Random Forest, and Support Vector Machines (SVM).
- **Dataset Management & Synthetic Generator**: Built-in generator for realistic synthetic GNSS trajectories with configurable spoofing attack profiles.
- **Secure Authentication**: JWT-based bearer authentication with bcrypt password hashing and user-isolated datasets/predictions.
- **History & Analytics**: Persistent SQLite storage tracking all past detection sessions, sequence-level predictions, and confidence distributions.

---

## 🛠️ Technologies Used

| Category | Technology / Library | Description |
|---|---|---|
| **Deep Learning & ML** | PyTorch / Torch 2.x | Neural network training & inference |
| | Scikit-Learn | Feature scaling, metrics calculation, baseline models |
| | Pandas & NumPy | Numerical computation, data wrangling, sequence generation |
| **Backend Framework** | FastAPI | High-performance Python async web framework |
| | Uvicorn | ASGI web server |
| | Pydantic v2 | Strict data validation and schema serialization |
| | SQLAlchemy | ORM database mapping |
| | python-jose & bcrypt | JWT token handling & password hashing |
| **Frontend Framework** | React 18 | Declarative reactive UI library |
| | Vite | Lightning-fast frontend build tooling |
| | Recharts | Interactive SVG charting for loss/accuracy curves |
| | Lucide React | Modern cybersecurity icon suite |
| | React Router DOM | Client-side routing across dashboard modules |
| **Database** | SQLite | Lightweight relational database storage |

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Client ["Client (Frontend)"]
        UI["React 18 + Vite Dashboard"]
        AuthContext["Auth Context (JWT)"]
        Pages["Upload | Training | Detection | Analytics"]
    end

    subgraph API ["Backend API (FastAPI)"]
        Router["API Router (/api)"]
        AuthMiddleware["JWT Authentication Middleware"]
        UploadService["Dataset & File Validation"]
        PredictService["Inference Pipeline"]
    end

    subgraph MLPipeline ["Machine Learning Core"]
        FE["Feature Extraction & Sliding Window (L=20)"]
        Scaler["StandardScaler (fitted scaler.pkl)"]
        BiGRU["Bidirectional GRU Network (PyTorch)"]
        Baselines["Baseline Evaluators (LR, RF, SVM)"]
    end

    subgraph Storage ["Persistent Storage"]
        DB[(SQLite: gnss_detection.db)]
        Models[("Trained Models (/trained_models)")]
        Datasets[("Datasets (/datasets)")]
    end

    UI --> Router
    Router --> AuthMiddleware
    AuthMiddleware --> DB
    Router --> UploadService
    UploadService --> Datasets
    UploadService --> DB
    Router --> PredictService
    PredictService --> FE
    FE --> Scaler
    Scaler --> BiGRU
    BiGRU --> DB
    PredictService --> UI
```

---

## 🧠 ML Model Details

### Architecture & Layer Structure

The deep learning model is implemented in PyTorch (`app/ml/bigru_model.py`):

1. **Input Layer**: `(batch_size, sequence_length=20, n_features)`
2. **Bidirectional GRU Layer 1**:
   - Hidden units: 64 per direction (Total output dimension: 128)
   - `batch_first=True`, `bidirectional=True`
3. **Dropout Layer 1**: Rate = `0.2` (prevents co-adaptation of features)
4. **Bidirectional GRU Layer 2**:
   - Hidden units: 32 per direction (Total output dimension: 64)
   - Final hidden state extraction
5. **Dropout Layer 2**: Rate = `0.2`
6. **Fully Connected Dense Layer**: 32 units with ReLU activation
7. **Classification Output Layer**: 1 unit with Sigmoid activation ($P(\text{Spoofed}) \in [0, 1]$)

### Label Encoding
- `0`: **Normal** GNSS Signal
- `1`: **Spoofed** GNSS Signal

### Training Strategy & Regularization
- **Optimizer**: Adam ($\text{learning rate} = 0.001$)
- **Loss Function**: Binary Cross-Entropy Loss (`BCELoss`)
- **Batch Size**: 32
- **Epochs**: Up to 25 with Early Stopping (patience: 4 epochs)
- **Learning Rate Scheduling**: ReduceLROnPlateau (factor: 0.5, patience: 5)
- **Data Splits**: 65% Training, 20% Validation, 15% Hold-out Testing

---

## 📋 Dataset Description

The system processes sequential GNSS time-series logs in CSV format.

### Supported Feature Schema

| Column Name | Data Type | Required | Description |
|---|---|---|---|
| `label` | `int` | **Yes** | Ground truth label: `0` (Normal), `1` (Spoofed) |
| `timestamp` | `float` | No | Epoch timestamp in seconds |
| `pseudorange` | `float` | Recommended | Measured pseudorange distance (meters) |
| `carrier_phase` | `float` | Recommended | Carrier phase observation (cycles) |
| `cn0` | `float` | Recommended | Carrier-to-Noise density ratio ($C/N_0$ in dB-Hz) |
| `doppler` | `float` | Recommended | Doppler frequency shift (Hz) |
| `elevation` | `float` | Recommended | Satellite elevation angle (degrees) |
| `azimuth` | `float` | Recommended | Satellite azimuth angle (degrees) |
| `latitude` | `float` | Recommended | Receiver estimated latitude (deg) |
| `longitude` | `float` | Recommended | Receiver estimated longitude (deg) |
| `altitude` | `float` | Recommended | Receiver estimated altitude (meters) |
| `velocity` | `float` | Recommended | Receiver estimated velocity (m/s) |

*Note: The dataset must have at least the `label` column. Missing feature columns are automatically imputed with median values during preprocessing.*

---

## 📁 Folder Structure

```
GNSS-Spoofing-Detection/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py                 # Central hyperparameters & paths
│   │   ├── main.py                   # FastAPI application initialization & routes
│   │   ├── api/                      # REST API endpoint routers
│   │   │   ├── auth.py               # User registration, login, JWT issuance
│   │   │   ├── dashboard.py          # Dashboard summary metrics
│   │   │   ├── dataset.py            # Upload, inspect, template download
│   │   │   ├── prediction.py         # Bi-GRU batch & sequence inference
│   │   │   └── training.py           # Training triggers, progress, & metrics
│   │   ├── database/
│   │   │   ├── db.py                 # SQLAlchemy engine & session setup
│   │   │   └── models.py             # User, Dataset, TrainingRun, Prediction ORM
│   │   ├── ml/                       # Machine learning core
│   │   │   ├── baseline_models.py    # LR, Random Forest, SVM benchmarks
│   │   │   ├── bigru_model.py        # PyTorch Bi-GRU neural network definition
│   │   │   ├── feature_engineering.py# Temporal diffs, rolling stats, C/N0 features
│   │   │   ├── predictor.py          # Inference runner & anomaly thresholding
│   │   │   ├── preprocessor.py       # Sliding window generation & data cleaning
│   │   │   ├── sklearn_compat.py     # Compatibility wrapper
│   │   │   └── trainer.py            # Model training loop & checkpointing
│   │   └── utils/
│   │       ├── security.py           # Bcrypt hashing & JWT verification
│   │       └── validators.py         # CSV format & column validation
│   ├── datasets/
│   │   ├── generate_synthetic.py     # Script to generate synthetic GNSS data
│   │   ├── gnss_sample_template.csv  # Minimal sample template for users
│   │   └── synthetic_gnss_dataset.csv# Pre-generated benchmark dataset (~5000 rows)
│   ├── trained_models/
│   │   ├── bigru_model.pt            # Pre-trained PyTorch Bi-GRU weights
│   │   ├── feature_config.json       # Feature column mappings
│   │   ├── model_metadata.json       # Benchmark metrics, confusion matrix, history
│   │   ├── scaler.pkl                # Fitted StandardScaler object
│   │   └── training_status.json      # Live training status tracking
│   ├── uploads/
│   │   └── .gitkeep                  # Working directory for uploaded files
│   ├── .env.example                  # Environment variable template
│   ├── create_demo_user.py           # Seed script for demo user account
│   ├── requirements.txt              # Backend Python dependencies
│   ├── run.py                        # Uvicorn launcher script
│   └── train.py                      # Standalone CLI training script
├── frontend/
│   ├── src/
│   │   ├── charts/                   # Recharts visualization modules
│   │   ├── components/               # UI components (Sidebar, Navbar, Card, etc.)
│   │   ├── context/                  # React Auth context
│   │   ├── pages/                    # 8 Application pages
│   │   │   ├── AboutPage.jsx         # Architecture & algorithm description
│   │   │   ├── DashboardPage.jsx     # Main cybersecurity dashboard
│   │   │   ├── DatasetUploadPage.jsx # CSV upload & preview
│   │   │   ├── DetectionPage.jsx     # Live spoofing detection interface
│   │   │   ├── EvaluationPage.jsx    # Model metrics & baseline comparison
│   │   │   ├── LoginPage.jsx         # User login
│   │   │   ├── ModelTrainingPage.jsx # Model retraining & progress monitor
│   │   │   ├── PredictionHistoryPage.jsx # History of detection runs
│   │   │   └── RegisterPage.jsx      # New account registration
│   │   ├── services/
│   │   │   └── api.js                # Axios HTTP client configured for backend
│   │   ├── App.jsx                   # React root router
│   │   ├── index.css                 # Global CSS styles
│   │   └── main.jsx                  # React DOM root entry
│   ├── index.html                    # HTML shell
│   ├── package.json                  # Frontend dependencies & scripts
│   ├── package-lock.json             # Locked dependency versions
│   └── vite.config.js                # Vite development and build configuration
├── .gitignore                        # Git exclusion rules
├── PROJECT_DOCUMENTATION.md          # Full academic & technical documentation
├── README.md                         # Project documentation
├── start_backend.bat                 # One-click Windows launch for backend
└── start_frontend.bat                # One-click Windows launch for frontend
```

---

## 🚀 How to Run the Project

### Prerequisites
- **Python**: Version 3.10, 3.11, or 3.12
- **Node.js**: Version 18.x or 20.x LTS
- **npm**: Version 9.x or higher

---

### Method 1: Quick Start (Windows Bat Scripts)

1. **Launch Backend**:
   Double click `start_backend.bat` or run:
   ```cmd
   start_backend.bat
   ```
2. **Launch Frontend**:
   Double click `start_frontend.bat` or run:
   ```cmd
   start_frontend.bat
   ```
3. Open your browser at **http://localhost:5173**.

---

### Method 2: Manual Step-by-Step Setup

#### Step 1: Backend Setup

```bash
# Navigate to the backend directory
cd backend

# Create a Python virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt

# Create demo account (demo@gnss.edu / demo1234)
python create_demo_user.py

# (Optional) Generate or regenerate the synthetic dataset
python datasets/generate_synthetic.py

# Start the FastAPI backend server
python run.py
```

The backend server will run on **http://localhost:8000**.  
Interactive Swagger API documentation: **http://localhost:8000/docs**.

#### Step 2: Frontend Setup

Open a new terminal window:

```bash
# Navigate to the frontend directory
cd frontend

# Install Node dependencies
npm install

# Start the Vite development server
npm run dev
```

The frontend will be accessible on **http://localhost:5173**.

---

## 🔑 Default Credentials

A demo account can be created automatically using `python create_demo_user.py`:
- **Email**: `demo@gnss.edu`
- **Password**: `demo1234`

Alternatively, you can register a new account on the `/register` page.

---

## 🌐 API Information

FastAPI provides an interactive OpenAPI interface available at `http://localhost:8000/docs`.

### Key Endpoints

| Category | Method | Endpoint | Description |
|---|---|---|---|
| **Auth** | `POST` | `/api/auth/register` | Register a new user |
| | `POST` | `/api/auth/login` | Login and obtain JWT access token |
| | `GET` | `/api/auth/me` | Fetch authenticated user profile |
| **Dataset** | `POST` | `/api/dataset/upload` | Upload and validate GNSS CSV dataset |
| | `GET` | `/api/dataset/list` | List all uploaded datasets for user |
| | `GET` | `/api/dataset/{id}` | Inspect rows and column distribution |
| | `GET` | `/api/dataset/sample-template` | Download sample CSV template |
| **Training** | `POST` | `/api/model/train` | Trigger asynchronous Bi-GRU training run |
| | `GET` | `/api/model/status` | Polling endpoint for training progress |
| | `GET` | `/api/model/metrics` | Fetch latest evaluation metrics & baselines |
| **Prediction** | `POST` | `/api/predict` | Upload GNSS CSV for spoofing classification |
| | `GET` | `/api/predictions` | Query historical prediction records |
| **Dashboard**| `GET` | `/api/dashboard/summary` | Global summary stats (accuracy, counts, runs) |

---

## 💡 Example Usage

### 1. Running Inference via Python API

```python
import requests

# 1. Authenticate
auth_resp = requests.post(
    "http://localhost:8000/api/auth/login",
    json={"email": "demo@gnss.edu", "password": "demo1234"}
)
token = auth_resp.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. Upload file for spoofing detection
with open("datasets/gnss_sample_template.csv", "rb") as f:
    files = {"file": ("gnss_sample.csv", f, "text/csv")}
    pred_resp = requests.post(
        "http://localhost:8000/api/predict",
        headers=headers,
        files=files
    )

result = pred_resp.json()
print("Detection Result:", result["overall_prediction"])  # NORMAL or SPOOFED
print("Confidence:", f"{result['confidence']:.2f}%")
print("Spoofed Sequence Ratio:", f"{result['spoofed_percentage']:.2f}%")
```

### 2. Retraining via Command Line

```bash
cd backend
python train.py --dataset datasets/synthetic_gnss_dataset.csv
```

---

## 📊 Model Performance & Evaluation

The pre-trained Bi-GRU model exhibits outstanding temporal classification capability compared to standard shallow classifiers:

| Model Architecture | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~78.4% | 0.76 | 0.79 | 0.77 | 0.85 |
| Random Forest | ~92.1% | 0.91 | 0.93 | 0.92 | 0.96 |
| Support Vector Machine (SVM) | ~89.6% | 0.88 | 0.91 | 0.89 | 0.94 |
| **Bi-GRU Network (Ours)** | **~98.5%** | **0.98** | **0.99** | **0.98** | **0.99** |

*All metrics are verified using stratified test splits and saved in `trained_models/model_metadata.json`.*

---

## 🔮 Future Enhancements

- [ ] **Hardware SDR Integration**: Real-time Software-Defined Radio (HackRF One, RTL-SDR, USRP) stream ingestion.
- [ ] **Multi-Constellation Fusion**: Cross-validation across GPS, GLONASS, Galileo, and BeiDou signals.
- [ ] **Transformer / Attention Mechanisms**: Evaluation of Temporal Convolutional Networks (TCN) and Vision Transformers on spectrograms.
- [ ] **Edge Deployment**: Quantization (INT8/FP16) for micro-deployments on NVIDIA Jetson Orin and Raspberry Pi 5.
- [ ] **Anti-Jamming & Meaconing Defense**: Expanding the detection taxonomy to jamming, meaconing, and multipath mitigation.

---

## 📄 License

This project is developed for academic research and educational purposes. Distributed under the MIT License.
