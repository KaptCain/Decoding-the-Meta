
import requests
import json

# Define the API endpoint and parameters



# Set custom headers (REQUIRED by Liquipedia)
headers = {
    'User-Agent': 'ValorantDataCollector/1.0 (https://github.com/KaptCain/Decoding-the-Meta; chase.collins120@gmail.com)',
    'Accept-Encoding': 'gzip'
}
s = requests.Session()
api_url = "https://liquipedia.net/valorant/api.php"
params = {
    'action': 'parse',
    'page': 'VCT/2024/Game_Changers_Championship',  # Replace with the page title you want
    'format': 'json'
}
# Make the API request
response = s.get(api_url, params=params)
print(response)
# Check if the request was successful
if response.status_code == 200:
    data = response.json()  # Convert response to JSON

    # Save data to a file
    with open("liquipedia_data.json", "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print("Data successfully saved to liquipedia_data.json")
else:
    print(f" Error: Received status code {response.status_code}")