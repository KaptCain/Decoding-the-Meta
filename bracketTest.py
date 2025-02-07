import json
from bs4 import BeautifulSoup

# Load JSON file containing HTML response
with open("liquipedia_data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Extract raw HTML
html_content = data["parse"]["text"]["*"]

# Parse HTML using BeautifulSoup
soup = BeautifulSoup(html_content, "html.parser")

tournament_data = []

for match in soup.find_all("div", class_="brkts-popup brkts-match-info-popup"):
    match_rounds = []

    # Extract match teams
    left_team = match.find("div", class_="brkts-popup-header-opponent brkts-popup-header-opponent-left")
    right_team = match.find("div", class_="brkts-popup-header-opponent brkts-popup-header-opponent-right")
    
    left_team_name = left_team.get_text(strip=True) if left_team else "Unknown"
    right_team_name = right_team.get_text(strip=True) if right_team else "Unknown"

    # Extract all match elements (win/loss indicators, agents, maps)
    elements = match.find_all(["i", "div"], class_=[
        "fas fa-check forest-green-text",  # Win Indicator
        "fas fa-times cinnabar-text",      # Loss Indicator
        "brkts-popup-body-element-thumbs brkts-champion-icon hide-mobile",  # Agents
        "brkts-popup-spaced"  # Map Name
    ])

    round_data = {}
    phase = 0  # Tracks which part of the sequence we're in

    for element in elements:
        # Phase 0: First team's win/loss indicator
        if phase == 0:
            if "fa-check" in element.get("class", []):
                round_data["team_1_winner"] = left_team_name
                round_data["team_2_winner"] = right_team_name
            else:
                round_data["team_1_winner"] = right_team_name
                round_data["team_2_winner"] = left_team_name
            phase += 1
        
        # Phase 1: First team's agents
        elif phase == 1:
            round_data["team_1_agents"] = [a["title"] for a in element.find_all("a") if "title" in a.attrs]  # Fix: Only get elements that have "title"
            phase += 1

        # Phase 2: Map name
        elif phase == 2:
            round_data["map"] = element.get_text(strip=True)
            phase += 1

        # Phase 3: Second team's win/loss indicator
        elif phase == 3:
            if "fa-check" in element.get("class", []):
                round_data["team_1_winner"] = right_team_name
                round_data["team_2_winner"] = left_team_name
            else:
                round_data["team_1_winner"] = left_team_name
                round_data["team_2_winner"] = right_team_name
            phase += 1
        
        # Phase 4: Second team's agents
        elif phase == 4:
            round_data["team_2_agents"] = [a["title"] for a in element.find_all("a") if "title" in a.attrs]  # Fix: Only get elements that have "title"
            phase = 0  # Reset for next round
            match_rounds.append(round_data)  # Store the completed round
            round_data = {}  # Reset for next round

    # Store match data with all rounds
    tournament_data.append({
        "left_team": left_team_name,
        "right_team": right_team_name,
        "rounds": match_rounds
    })

# Save extracted tournament data
with open("extracted_valorant_tournament5.json", "w", encoding="utf-8") as output_file:
    json.dump(tournament_data, output_file, indent=4)

print("✅ Extracted tournament round data saved successfully!")
