# import requests
# import json

# # API Endpoint
# MATCHES_URL = "https://vlr.orlandomm.net/api/v1/matches"

# # Step 1: Make an API Request
# response = requests.get(MATCHES_URL)

# # Step 2: Check if request was successful
# if response.status_code == 200:
#     matches = response.json()  # Convert response to JSON format
#     print(json.dumps(matches, indent=4))  # Pretty-print JSON response
# else:
#     print("Failed to fetch data:", response.status_code)
import requests
import json

# API Endpoint
# MATCHES_URL = "https://vlr.orlandomm.net/api/v1/matches"

# # Step 1: Make an API Request
# response = requests.get(MATCHES_URL)

# # Step 2: Check if request was successful
# if response.status_code == 200:
#     matches = response.json()  # Convert response to JSON format
    
#     # Step 3: Save data to a file
#     with open("matches.json", "w", encoding="utf-8") as file:
#         json.dump(matches, file, indent=4)  # Save as formatted JSON
    
#     print("Match data successfully saved to matches.json")
# else:
#     print("Failed to fetch data:", response.status_code)


#API Endpoint
events_URL = "https://vlr.orlandomm.net/api/v1/players?limit=all&country=all&agent=omen"

# Step 1: Make an API Request
response = requests.get(events_URL)

# Step 2: Check if request was successful
if response.status_code == 200:
    matches = response.json()  # Convert response to JSON format
    
    # Step 3: Save data to a file
    with open("players.json", "w", encoding="utf-8") as file:
        json.dump(matches, file, indent=4)  # Save as formatted JSON
    
    print("Match data successfully saved to players.json")
else:
    print("Failed to fetch data:", response.status_code)