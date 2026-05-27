# RoadWatch AI 🛣️🤖

**RoadWatch AI** is a comprehensive, citizen-empowering platform built to monitor road quality, ensure budget transparency, and hold authorities and contractors accountable.

It fulfills all the requirements set forth in the Hackathon **Section 1.2 Rubric**, providing an AI-powered chatbot and dashboard to track public spending and infrastructure health.

## 🚀 Key Features

### 1. Offline Functionality & Low-Network Robustness
- **Progressive Web App (PWA):** The frontend integrates a Service Worker (`sw.js`) that caches core assets (HTML, CSS, JS, Locales, Map Tiles), allowing the app to open instantly and function perfectly even when internet access is completely lost.
- **Local AI LLM:** Features a completely offline, privacy-first Local LLM (HuggingFace `SmolLM2-360M-Instruct`) running locally on your hardware. It analyzes your prompts and road data without requiring cloud APIs.

### 2. Location-Based Accountability
- **Contractor & Budget Transparency:** Clicking on the interactive map uses Reverse Geocoding to automatically identify the **Road Type**, **Contractor Name**, **Last Relaying Date**, and **Budget (Sanctioned vs. Spent)**.
- **Smart Complaint Routing:** Using an internal Authority Map, any complaints you file are automatically routed to the correct executive engineer, block development officer, or regional NHAI office depending on the road tier.

### 3. Global Applicability
- **International Fallbacks:** Not restricted to India! RoadWatch's backend intelligently detects non-Indian locations via OpenStreetMap tags (`motorway`, `trunk`, `residential`, etc.) and seamlessly maps them to universal authority equivalents (e.g., "National Transport Authority", "Municipal Corporation").

### 4. RAG AI Chatbot with Native Localization
- Ask questions directly to the offline LLM about budgets, contractor accountability, and road laws.
- **Language Detection:** Using `langdetect`, the system parses queries in English, Hindi, Bengali, Tamil, Telugu, and Marathi. It uses a custom transliteration algorithm to allow users to ask questions in regional languages using English alphabets (e.g. "Mera road kharab hai"), preserving cultural context!

### 5. Advanced Machine Learning Diagnostics
- Includes 4 predictive ML models trained on PMGSY/OMMAS data:
  - **Budget Anomaly Detector:** Flags potential fund misuse by comparing expenditure against observed quality.
  - **Complaint Severity Classifier:** Estimates SLA requirements and categorizes urgency based on proximity to schools/hospitals.
  - **Lifecycle Predictor:** Forecasts when a road will fail based on its construction material and traffic class.

## 📦 Data Sources
All data integrated into this application is derived exclusively from publicly available resources. No paid APIs are used.
- **OMMAS (pmgsy.dord.gov.in)**: Pradhan Mantri Gram Sadak Yojana reports for rural roads, sanctions, and contractor grading.
- **data.gov.in**: Government budget allocations and infrastructure expenditure.
- **OpenStreetMap (Nominatim)**: Free, open-source reverse geocoding for precise map-to-address conversion worldwide.

## 🛠️ How to Run Locally

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the Web Server:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
3. **Access the App:**
   Open your browser and navigate to `http://localhost:8000`.

*(Note: The first time you ask the AI Chatbot a question, it will download the offline model (~700MB) directly to your machine. After this initial setup, it will run entirely offline!)*

## 🛡️ Security
This platform includes explicit protections against prompt injection and Cross-Site Scripting (XSS). All inputs are sanitized before being passed to the backend, rate-limiting is enforced globally per IP, and sensitive AI data is constrained strictly via system prompts.

---
*Built for the National Road Safety Hackathon 2026 | CoERS, IIT Madras*
