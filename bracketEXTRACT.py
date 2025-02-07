import json
import re
from bs4 import BeautifulSoup

# Load JSON data from the file
with open("liquipedia_data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Extract raw HTML from the JSON response
html_content = data["parse"]["text"]["*"]

# Parse HTML using BeautifulSoup
soup = BeautifulSoup(html_content, "html.parser")

# Extract all match brackets
match_data = []

# Find all match containers (update tag structure based on actual HTML layout)
for match in soup.find_all("div", class_="brkts-popup brkts-match-info-popup"):
    left_team = [team.get_text(strip=True) for team in match.find_all("div", class_="brkts-popup-header-opponent brkts-popup-header-opponent-left")]
    right_team = [team.get_text(strip=True) for team in match.find_all("div", class_="brkts-popup-header-opponent brkts-popup-header-opponent-right")]
    maps = [map_.get_text(strip=True) for map_ in match.find_all("div", class_="brkts-popup-spaced")]
    
    
    left_agents = [agent.get_text(strip=True) for agent in match.find_all("div", class_="brkts-popup-body-element-thumbs brkts-champion-icon hide-mobile")]
    
    left_agent_divs = match.find_all("div", class_="brkts-popup-body-element-thumbs brkts-champion-icon hide-mobile")# Finding all divs containing agents
    
    left_agents = []
    print(f"Found agent divs: {len(left_agent_divs)}")
    for agent_div in left_agent_divs:
        agent_links = agent_div.find_all("a")  # Get all <a> tags inside this div
        for agent in agent_links:
            agent_name = agent.get("title")  # Extracts "Viper", "Brimstone", etc.
            left_agents.append(agent_name)
    print(f"Agents found for this match: {left_agents}")
    right_agents = [agent.get_text(strip=True) for agent in match.find_all("div", class_="brkts-popup-body-element-thumbs brkts-champion-icon brkts-popup-body-element-thumbs-right hide-mobile")]
    right_agent_divs = match.find_all("div", class_="brkts-popup-body-element-thumbs brkts-champion-icon brkts-popup-body-element-thumbs-right hide-mobile")# Finding all divs containing agents
    right_agents = []
    print(f"Found agent divs: {len(right_agent_divs)}")
    for agent_div in right_agent_divs:
        agent_links = agent_div.find_all("a")  # Get all <a> tags inside this div
        for agent in agent_links:
            agent_name = agent.get("title")  # Extracts "Viper", "Brimstone", etc.
            right_agents.append(agent_name)

    
    winner = match.find("div", class_="brkts-popup-spaced brkts-popup-winloss-icon").get_text(strip=True) if match.find("div", class_="brkts-popup-spaced brkts-popup-winloss-icon") else "Unknown"

    match_data.append({
        "left team": left_team,
        "right team": right_team,
        "maps": maps,
        "left agents": left_agents,
        "right agents": right_agents,
        "winner": winner
    })

# Save cleaned tournament bracket data to a JSON file
with open("valorant_tournament_bracket.json", "w", encoding="utf-8") as output_file:
    json.dump(match_data, output_file, indent=4)

print("✅ Tournament bracket data saved to 'valorant_tournament_bracket.json'")




