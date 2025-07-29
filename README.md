# 🔧 RAG-Enabled Predictive Maintenance & Troubleshooting Assistant

### 🎓 Final Year Major Project — Artificial Intelligence and Data Science Engineering

---

## 📌 Overview

This project builds a **RAG-enabled generative AI assistant** that performs two key tasks:
1. **Predicts machine failures** using sensor data via machine learning models
2. **Suggests troubleshooting steps** using **Retrieval-Augmented Generation (RAG)** to search maintenance manuals and historical repair logs

The goal is to simulate a real-world AI assistant that aids technicians and engineers in **predictive maintenance**, minimizing downtime and improving reliability.

---

## 🧠 Motivation

In industrial environments, unexpected machine failures can lead to costly downtimes. Traditional maintenance methods rely heavily on manual inspection and static documentation.

This project uses **machine learning for proactive failure prediction** and **LLMs with RAG** to extract relevant repair procedures from documents—bringing the power of AI to intelligent, context-aware maintenance systems.

---

## 🏭 Use Case: Turbofan Engine (NASA CMAPSS Dataset)

We focus on the **FD001 subset** of the CMAPSS dataset from NASA, simulating a jet engine's multivariate sensor data over time. The assistant predicts:
- Remaining Useful Life (RUL)
- Type of impending component failure

It then retrieves context from:
- Maintenance manuals (PDF/Text)
- Simulated repair logs

---

## ⚙️ Project Architecture

            +----------------------+
            |  Sensor Time Series  |
            +----------------------+
                      |
                      v
    +-------------------------------+
    |   Predictive Maintenance ML   |
    |  (RUL Prediction / Failure ID)|
    +-------------------------------+
                      |
                      v
  +---------------------------------------+
  | LLM-based Troubleshooting Assistant   |
  | (using RAG: Retriever + Generator)    |
  +---------------------------------------+
                      |
                      v
       +-------------------------------+
       | Suggested Fixes & Instructions |
       +-------------------------------+

---

## 📂 Folder Structure

RAG-Predictive-Maintenance/
│
├── data/ # Sensor data, manuals, repair logs
├── notebooks/ # EDA, training, RAG experiments
├── models/ # Trained ML models (RUL/classifier)
├── rag_pipeline/ # Embedding, retriever, LLM code
├── src/ # Core preprocessing & ML code
├── ui/ # Streamlit or Flask frontend
├── requirements/ requirements.txt # Python dependencies
├── README.md # You're here
└── presentation/ # PPT, architecture diagram, etc.\


---

## 🧪 Components

### ✅ 1. Predictive Maintenance
- Dataset: [NASA CMAPSS FD001](https://data.nasa.gov/dataset/CMAPSS-Data/njnj-t8ks)
- ML Models: LSTM for RUL prediction, XGBoost for fault classification
- Metrics: RMSE (RUL), Accuracy/F1 (faults)

### ✅ 2. RAG Pipeline
- Embedding: `all-MiniLM-L6-v2`, `Instructor-XL` (HuggingFace)
- Vector DB: `FAISS` (local), or `Chroma`
- Retriever: `LangChain` + custom retriever
- Generator: GPT-4 (OpenAI) or open-source like Mistral

### ✅ 3. Frontend
- Streamlit web interface:
  - Upload sensor log
  - View predicted failure/RUL
  - Receive repair recommendations with manual excerpts

---

## 📊 Sample Input/Output

### Input:

{
  "sensor_data": "engine_124.csv"
}
### Output:

🔧 Prediction: High-Pressure Compressor Degradation
📉 RUL: 42 cycles
📖 Troubleshooting Steps:
1. Shut off fuel line (Manual Sec 4.3)
2. Inspect HPT vane alignment
3. Replace corroded blade tips if >2mm wear

| Resource                                                              | Description                               |
| --------------------------------------------------------------------- | ----------------------------------------- |
| [CMAPSS Dataset](https://data.nasa.gov/dataset/CMAPSS-Data/njnj-t8ks) | Turbofan sensor data                      |
| Maintenance Manuals                                                   | Sample PDFs (added in `/data/manuals`)    |
| Simulated Logs                                                        | Created for testing RAG with repair notes |


| Area  | Tools/Libraries                            |
| ----- | ------------------------------------------ |
| ML    | scikit-learn, PyTorch, XGBoost             |
| RAG   | LangChain, FAISS, HuggingFace Transformers |
| LLM   | GPT-4, Mistral-7B (via HuggingFace)        |
| UI    | Streamlit                                  |
| Other | pandas, matplotlib, sentence-transformers  |

🧠 Team
Names: 
Zidan Shaikh
Salif Shaikh
Dhruv Vaidya
Vishwa Pawar
Department: Artificial Intelligence and Data Science

Institution: Thadomal Shahani Engineering College

Mentor: Prof. Dr. GT Thampi