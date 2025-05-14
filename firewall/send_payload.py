import pandas as pd
import random
import requests
import json

# Load Dataset
data = pd.read_csv("C:/Users/rajes/OneDrive/Desktop/gans_firewall/dataset/SQLInjection_XSS_MixDataset.1.0.0.csv")  # Replace with your actual dataset filename

# Pick a random row
#random_entry = data.sample(n=1).iloc[0]
#random_sentence = random_entry["Sentence"]  # Adjust column name if necessary
random_sentence = data["Sentence"].sample(n=1).values[0]
# Define Firewall API Endpoint
firewall_url = "http://127.0.0.1:5000/detect"  # Change this if your API is hosted elsewhere

# Prepare the request payload
payload = {"sentence": random_sentence}

# Send request to the firewall
response = requests.post(firewall_url, json=payload, headers={"Content-Type": "application/json"})

# Create JSON Request
json_request = {
    "sentence": random_sentence
}

# Convert to JSON string (for API request)
json_string = json.dumps(json_request, indent=4)


# Generate cURL Command
curl_command = f"""curl --location --request POST '{firewall_url}' \
--header 'Content-Type: application/json' \
--data-raw '{{"sentence": "{random_sentence}"}}'"""

# Print Response
print("Sent Sentence:", random_sentence)
print("Firewall Response:", response.json())
#print(curl_command)

print("\nGenerated JSON Request:")
print(json_string)


curl_command = f"""curl --location --request POST '{firewall_url}' \
--header 'Content-Type: application/json' \
--data-raw '{json_string}'"""


print(curl_command)
