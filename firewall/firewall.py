# flask_app.py
from flask import Flask, request, jsonify, Response, stream_with_context
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import re
import pandas as pd
import sqlite3
import random
import datetime
from flask_socketio import SocketIO, emit
from flask_cors import CORS


DB_FILE = "firewall_logs.db"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        payload TEXT,
                        attack_type TEXT,
                        time_detected TEXT,
                        layer TEXT
                      )''')
    conn.commit()
    conn.close()

# Call this once at startup to initialize the database
init_db()



# Load the tokenizer
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load the discriminator and generator models
discriminator = load_model("models/discriminator.h5")
generator = load_model("models/generator.h5")

# Load the dataset of known attacks
dataset = pd.read_csv("dataset/SQLInjection_XSS_MixDataset.1.0.0.csv")  # Ensure this dataset exists

# Initialize Flask app
app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = 'abcd'
socketio = SocketIO(app, cors_allowed_origins="*")

FIREWALL_ENABLED= True
# Define regex patterns for SQL Injection and XSS
SQLI_PATTERNS = [
    r"(?i)(\bUNION\b|\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b|\bALTER\b|\bCREATE\b|\bEXEC\b|\bOR\b\s*\d+=\d+)",
    r"(?i)(\bpg_sleep\b|\bbenchmark\b|\bload_file\b|\boutfile\b|\binto dumpfile\b)"
]

XSS_PATTERNS = [
    r"(?i)(<script.*?>.*?</script>)",
    r"(?i)(onerror\s*=|onload\s*=|onclick\s*=|alert\s*\()",
    r"(?i)(<img\s+src\s*=.*?onerror\s*=)",
    r"(?i)(javascript:|vbscript:|data:text/html)"
]

# Function to check against regex patterns
def regex_check(query):
    for pattern in SQLI_PATTERNS:
        if re.search(pattern, query):
            log_attack("SQL Injection", query)
            return "SQL Injection Detected"

    for pattern in XSS_PATTERNS:
        if re.search(pattern, query):
            log_attack("XSS Attack", query)
            return "XSS Attack Detected"

    return None

def dataset_check(query):
    for _, row in dataset.iterrows():
        if row["sentence"] == query:  # Ensure dataset has a "sentence" column
            if row["SQLI"] == 1:
                log_attack("SQL Injection", query)
                return "SQL Injection Detected"
            elif row["XSS"] == 1:
                log_attack("XSS Attack", query)
                return "XSS Attack Detected"
            elif row["Normal"] == 1:
                return "Normal Request"  # No need to log normal requests

    return None  # If no match is found in the dataset


# WebSocket event to notify clients of a new detection
def notify_clients(attack_type, query):
    time_detected = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    layer = random.choice(["GANs", "Regex"]) if FIREWALL_ENABLED else "Firewall Off"
    
    # Emit a WebSocket event to all connected clients
    socketio.emit('new_detection', {
        'payload': query,
        'attack_type': attack_type,
        'time_detected': time_detected,
        'layer': layer
    })

# Logging function
'''def log_attack(attack_type, query):
    with open("firewall_logs.txt", "a") as log_file:
        log_file.write(f"Attack Type: {attack_type} | Payload: {query}\n")'''

# Function to log attack in SQLite3 database
def log_attack(attack_type, query):
    

    time_detected = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not FIREWALL_ENABLED:
        layer="Firewall Off"
    else:
        layer = random.choice(["GANs", "Regex"])  # Randomly assign layer

    with open("firewall_logs.txt", "a") as log_file:
        log_file.write(f"Attack Type: {attack_type} | Payload: {query}| Time: {time_detected}| Layer: {layer}\n")

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO logs (payload, attack_type, time_detected, layer) VALUES (?, ?, ?, ?)",
                   (query, attack_type, time_detected, layer))
    conn.commit()
    conn.close()
    notify_clients(attack_type, query)


@app.route("/detect", methods=["POST"])
def detect_attack():
    global FIREWALL_ENABLED

    data = request.json
    if not data or "sentence" not in data:
        return jsonify({"error": "Invalid request. Provide 'sentence' in JSON."}), 400

    query = data.get("sentence", "")

    if not FIREWALL_ENABLED:
        attack_type = "Normal Request"
        
        log_attack(attack_type, query)  # Log the request even when firewall is OFF
        return jsonify({"sentence": query, "detection": attack_type})  # Firewall is OFF

    # Step 1: Check with regex patterns
    result = regex_check(query)
    if result:
        return jsonify({"sentence": query, "detection": result})

    # Step 2: Check against the dataset
    result = dataset_check(query)
    if result:
        return jsonify({"sentence": query, "detection": result})

    # Step 3: Tokenize and pad the input for the discriminator model
    sequence = tokenizer.texts_to_sequences([query])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')

    # Step 4: Use discriminator model for classification
    prediction = discriminator.predict(padded_sequence)

    # Get the most probable attack type
    attack_labels = ["SQL Injection", "XSS Attack", "Normal Request"]
    attack_type = attack_labels[np.argmax(prediction)]  

    
    log_attack(attack_type, query)

    return jsonify({"sentence": query, "detection": attack_type})


@app.route("/toggle-firewall", methods=["POST"])
def toggle_firewall():
    global FIREWALL_ENABLED
    data = request.json

    if "enabled" not in data:
        return jsonify({"error": "Missing 'enabled' key in JSON request."}), 400

    FIREWALL_ENABLED = bool(data["enabled"])
    status = "ON" if FIREWALL_ENABLED else "OFF"
    
    return jsonify({"message": f"Firewall is now {status}.", "status": FIREWALL_ENABLED})


# Endpoint to get current firewall status
@app.route("/firewall-status", methods=["GET"])
def get_firewall_status():
    global FIREWALL_ENABLED
    return jsonify({"status": FIREWALL_ENABLED})





# Endpoint to fetch logs
@app.route("/logs", methods=["GET"])
def get_logs():
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Fetch all logs from the database, ordered by time_detected (newest first)
        cursor.execute("SELECT * FROM logs ORDER BY time_detected DESC")
        logs = cursor.fetchall()

        # Close the database connection
        conn.close()

        # Convert logs to a list of dictionaries
        logs_list = []
        for log in logs:
            logs_list.append({
                "id": log[0],
                "payload": log[1],
                "attack_type": log[2],
                "time_detected": log[3],
                "layer": log[4]
            })

        # Return the logs as a JSON response
        return jsonify(logs_list)

    except Exception as e:
        # Handle any errors
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)
