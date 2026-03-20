"""
Script automatique complet : Récupère les emails Gmail et lance l'interface web
AVEC MODÈLE TF-IDF - SEUILS : 0-80% = SAFE, 81-100% = SUSPECT
AMÉLIORATION : Réduit les faux positifs pour emails sans URL
"""

from flask import Flask, send_file, jsonify
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import os
import base64
import json
import webbrowser
import threading
import time
import re

app = Flask(__name__)
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# ==================== CHARGEMENT DU MODÈLE IA ====================

print("\n🤖 Chargement du modèle IA...")

# Variables globales pour le modèle
PIPELINE = None
TFIDF = None
CLF = None
THRESHOLD = 0.5

try:
    import joblib
    import numpy as np
    
    print("\n📂 Recherche du modèle...")
    
    # Chercher le fichier phishing_pipeline_ultra.pkl
    model_path = r"phishing_pipeline_ultra.pkl"
    
    if not os.path.exists(model_path):
        # Essayer dans le dossier courant
        model_path = "phishing_pipeline_ultra.pkl"
        
    if not os.path.exists(model_path):
        print("   ❌ Fichier 'phishing_pipeline_ultra.pkl' introuvable !")
        print("\n   💡 SOLUTION :")
        print("   1. Lancez : python train_model_ultra.py")
        print("   2. Puis relancez : python simple_server.py")
        raise FileNotFoundError("Modèle introuvable")
    
    print(f"   🔄 Chargement depuis : {model_path}")
    PIPELINE = joblib.load(model_path)
    
    # Extraire les composants du pipeline
    TFIDF = PIPELINE['tfidf']
    CLF = PIPELINE['clf']
    THRESHOLD = PIPELINE['threshold']
    
    print(f"   ✅ Modèle chargé avec succès !")
    print(f"   🎯 Seuil optimal : {THRESHOLD:.2f}")
    print("   🎯 Modèle IA prêt à analyser les emails !")
    
except Exception as e:
    print(f"\n   ⚠️  Erreur lors du chargement : {e}")
    print(f"   ℹ️  Type d'erreur : {type(e).__name__}")
    print("   ❌ Le modèle doit être chargé pour fonctionner !")
    import traceback
    traceback.print_exc()
    exit(1)

# ==================== FONCTIONS DE NETTOYAGE ====================

def clean_text(text):
    """Nettoie le texte exactement comme dans l'entraînement"""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def has_url(text):
    """Vérifie si le texte contient une URL"""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}'
    return bool(re.search(url_pattern, text))

# ==================== PARTIE 1 : RÉCUPÉRATION DES EMAILS ====================

def get_gmail_service():
    """Connexion à Gmail"""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'token!!!!!!!',
                SCOPES
            )
            creds = flow.run_local_server(port=8000)
        
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return build('gmail', 'v1', credentials=creds)

def get_email_details(service, msg_id):
    """Récupère les détails complets d'un email"""
    try:
        message = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
        
        headers = message['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'Sans sujet')
        sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Inconnu')
        date = next((h['value'] for h in headers if h['name'].lower() == 'date'), '')
        
        # Récupérer le corps du message
        body = ''
        if 'parts' in message['payload']:
            for part in message['payload']['parts']:
                if part['mimeType'] == 'text/plain':
                    if 'data' in part['body']:
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                        break
        elif 'body' in message['payload'] and 'data' in message['payload']['body']:
            body = base64.urlsafe_b64decode(message['payload']['body']['data']).decode('utf-8', errors='ignore')
        
        if not body:
            body = "Aucun contenu texte disponible"
        
        return {
            'id': msg_id,
            'subject': subject,
            'sender': sender,
            'date': date,
            'body': body[:500]
        }
    except Exception as e:
        print(f"   ⚠️  Erreur email {msg_id}: {e}")
        return None

def calculate_phishing_score(subject, sender, body):
    """
    Calcul du score de phishing avec VOTRE MODÈLE TF-IDF
    AMÉLIORATION : Réduit le score si aucune URL n'est détectée
    """
    
    # Vérifier que le modèle est chargé
    if TFIDF is None or CLF is None:
        print(f"   ❌ Modèle non chargé ! Impossible d'analyser l'email.")
        return 0
    
    try:
        # Combiner subject, sender et body
        full_text = f"{subject} {sender} {body}"
        
        # Vérifier la présence d'URLs
        contains_url = has_url(full_text)
        
        # Nettoyer le texte
        cleaned_text = clean_text(full_text)
        
        # Transformer avec TF-IDF
        text_tfidf = TFIDF.transform([cleaned_text])
        
        # Prédire la probabilité
        prob = CLF.predict_proba(text_tfidf)[0][1]
        
        # Convertir en score de 0 à 100
        score = int(prob * 100)
        
        # ✨ AMÉLIORATION : Réduire le score si aucune URL
        if not contains_url:
            # Réduire le score de 30% pour les emails sans URL
            # Les phishing contiennent presque toujours des liens
            score = int(score * 0.7)
            
            # Plafonner à 60% maximum pour emails sans URL
            score = min(score, 60)
        
        return min(max(score, 0), 100)
        
    except Exception as e:
        print(f"   ⚠️  Erreur prédiction IA : {e}")
        import traceback
        traceback.print_exc()
        return 0

def fetch_emails_from_gmail(max_results=40):
    """Récupère les derniers emails depuis Gmail"""
    print("\n" + "="*60)
    print("📥 ÉTAPE 1 : RÉCUPÉRATION DES EMAILS")
    print("="*60)
    
    print("\n🔄 Connexion à Gmail...")
    service = get_gmail_service()
    
    print("📧 Récupération de la liste des emails...")
    results = service.users().messages().list(userId='me', maxResults=max_results).execute()
    messages = results.get('messages', [])
    
    emails = []
    total = len(messages)
    print(f"📨 Traitement de {total} emails avec l'IA TF-IDF...\n")
    
    for i, msg in enumerate(messages):
        print(f"   [{i+1}/{total}] Analyse IA en cours...", end='\r')
        email_data = get_email_details(service, msg['id'])
        if email_data:
            # Calculer le score de phishing AVEC VOTRE MODÈLE IA
            email_data['phishingScore'] = calculate_phishing_score(
                email_data['subject'],
                email_data['sender'],
                email_data['body']
            )
            emails.append(email_data)
    
    print(f"\n\n✅ {len(emails)} emails analysés avec succès!")
    
    # Sauvegarder dans le fichier JSON
    with open('emails_data.json', 'w', encoding='utf-8') as f:
        json.dump(emails, f, ensure_ascii=False, indent=2)
    
    print("💾 Données sauvegardées dans 'emails_data.json'")
    
    # Statistiques avec NOUVEAUX seuils : 0-80% = safe, 81-100% = suspect
    safe = len([e for e in emails if e['phishingScore'] <= 80])
    suspect = len([e for e in emails if e['phishingScore'] > 80])
    
    print(f"\n📊 Statistiques :")
    print(f"   🟢 Emails sûrs (0-80%) : {safe}")
    print(f"   🟠 Emails suspects (81-100%) : {suspect}")
    print(f"   💡 Emails sans URL : score réduit automatiquement")
    
    return True

# ==================== PARTIE 2 : SERVEUR WEB ====================

@app.route('/')
def index():
    """Page principale - Sert le fichier HTML"""
    return send_file('emails_viewer.html')

@app.route('/emails_data.json')
def get_emails():
    """API pour récupérer les emails"""
    try:
        with open('emails_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def open_browser():
    """Ouvre automatiquement le navigateur après 2 secondes"""
    time.sleep(2)
    print("\n🌐 Ouverture automatique du navigateur...")
    webbrowser.open('http://localhost:5000')

def start_web_server():
    """Lance le serveur Flask"""
    print("\n" + "="*60)
    print("🚀 ÉTAPE 2 : LANCEMENT DE L'INTERFACE WEB")
    print("="*60)
    print("\n📱 Serveur démarré sur : http://localhost:5000")
    print("💡 Appuyez sur Ctrl+C pour arrêter le serveur\n")
    print("="*60 + "\n")
    
    # Ouvrir le navigateur automatiquement
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Lancer Flask sans reloader pour éviter les doublons
    app.run(debug=False, port=5000, use_reloader=False)

# ==================== PROGRAMME PRINCIPAL ====================

if __name__ == '__main__':
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "📧 APPLICATION GMAIL" + " "*23 + "║")
    print("║" + " "*8 + "Seuils : 0-80% Safe | 81-100% Suspect" + " "*12 + "║")
    print("║" + " "*10 + "✨ Anti-faux positifs activé" + " "*18 + "║")
    print("╚" + "="*58 + "╝")
    
    try:
        # ÉTAPE 1 : Récupérer les emails
        success = fetch_emails_from_gmail(40)
        
        if success:
            # Vérifier que le fichier HTML existe
            if not os.path.exists('emails_viewer.html'):
                print("\n❌ ERREUR : Le fichier 'emails_viewer.html' est introuvable !")
                print("   Assurez-vous que le fichier HTML est dans le même dossier.")
                exit(1)
            
            # ÉTAPE 2 : Lancer l'interface web
            start_web_server()
        else:
            print("\n❌ Impossible de récupérer les emails.")
            print("   Vérifiez votre configuration Google Cloud Console.")
            
    except KeyboardInterrupt:
        print("\n\n👋 Arrêt du serveur. Au revoir !")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()