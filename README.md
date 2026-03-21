# Phishing Email Detector 🎣🔍

AI-powered application that detects phishing emails in your 
Gmail inbox using machine learning.

## 📋 Table of Contents
- [Features](#-features)
- [Built with](#-built-with)
- [Gmail API Setup](#-gmail-api-setup)
- [How to run](#-how-to-run)
- [Project Structure](#-project-structure)
- [How it works](#-how-it-works)
- [Security](#-security)
- [Author](#-author)

## ✨ Features
- Real-time Gmail inbox analysis
- AI phishing score for each email (0-100%)
- Automatic classification: 🟢 Safe (0-80%) / 🟠 Suspect (81-100%)
- Anti false-positive system for emails without URLs
- Web interface to visualize results

## 🛠 Built with
- Python
- Flask
- scikit-learn (TF-IDF + Logistic Regression)
- Gmail API
- joblib

## 🔑 Gmail API Setup
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project
3. Enable **Gmail API**
4. Go to **Credentials** → Create **OAuth 2.0 Client ID**
5. Download the credentials file and rename it to `credentials.json`
6. Place `credentials.json` in the project root folder
7. Run the app — it will ask you to login with Google the first time
8. A `token.pickle` file will be generated automatically

## 🚀 How to run
1. Clone the repo
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Setup Gmail API credentials (see Gmail API Setup section)
4. Run the app:
```bash
python simple_server.py
```
5. Open your browser at: `http://localhost:5000`

> ✅ The trained model `phishing_pipeline_ultra.pkl` is already
> included — no need to retrain!

> 💡 If you want to retrain the model with your own dataset:
> ```bash
> python train_model_ultra.py
> ```

## 📂 Project Structure
```
phishing-email-detector/
├── simple_server.py          # Main app + Gmail API
├── train_model_ultra.py      # Model training script
├── phishing_pipeline_ultra.pkl  # Trained model (ready to use)
├── emails_viewer.html        # Web interface
├── requirements.txt          # Dependencies
└── .gitignore
```

## ⚙️ How it works
1. Connects to your Gmail inbox via Google API
2. Fetches your last 40 emails
3. Cleans and analyzes each email with TF-IDF
4. Predicts a phishing score using Logistic Regression
5. Reduces score for emails without URLs (anti false-positive)
6. Displays results in a web interface

## 🔒 Security
- `credentials.json` and `token.pickle` are in `.gitignore`
- Never share these files
- The app only reads your emails — it never modifies or deletes them

## ⚠️ Important
Never commit these files to GitHub:
```
token.pickle
credentials.json
```

