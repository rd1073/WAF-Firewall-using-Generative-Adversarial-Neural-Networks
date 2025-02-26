from flask import Flask, request, jsonify
import re

app = Flask(__name__)

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

# Function to check incoming requests
def firewall_detect(query):
    for pattern in SQLI_PATTERNS:
        if re.search(pattern, query):
            log_attack("SQL Injection", query)
            return "SQL Injection Detected"

    for pattern in XSS_PATTERNS:
        if re.search(pattern, query):
            log_attack("XSS Attack", query)
            return "XSS Attack Detected"

    return "Normal Request"

# Logging function
def log_attack(attack_type, query):
    with open("firewall_logs.txt", "a") as log_file:
        log_file.write(f"Attack Type: {attack_type} | Payload: {query}\n")

# API Route for Firewall
@app.route("/firewall", methods=["POST"])
def firewall():
    data = request.json
    if not data or "query" not in data:
        return jsonify({"error": "Invalid request. Provide 'query' in JSON."}), 400

    query = data["query"]
    result = firewall_detect(query)
    return jsonify({"query": query, "detection": result})

# Run the firewall server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
