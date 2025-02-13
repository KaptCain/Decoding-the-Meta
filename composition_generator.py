from collections import defaultdict
import random
import pandas as pd
import sqlLoader
from sqlLoader import SQLdb
# List of all Valorant agents

df = SQLdb()

agents = [
    "Brimstone", "Phoenix", "Sage", "Sova", "Viper", "Cypher", "Reyna", "Killjoy", "Breach", "Omen",
    "Jett", "Raze", "Skye", "Yoru", "Astra", "KAY/O", "Chamber", "Neon", "Fade", "Harbor", "Gekko",
    "Deadlock", "Iso", "Clove", "Vyse", "Tejo"
]

# Given good team compositions
good_compositions = [
    ["Abyss", "Iso", "Clove", "Sova", "Yoru", "Vyse"],
    ["Abyss", "Cypher", "Clove", "Jett", "Deadlock", "Gekko"],
    ["Abyss", "Clove", "Sova", "Reyna", "Neon", "Deadlock"],
    ["Abyss", "Clove", "Sova", "Jett", "Tejo", "Deadlock"],
    ["Bind", "Clove", "Chamber", "Tejo", "Deadlock", "Raze"],
    ["Bind", "Clove", "Sage", "Reyna", "Jett", "Deadlock"],
    ["Bind", "Cypher", "Clove", "Skye", "Reyna", "Neon"],
    ["Bind", "Clove", "Skye", "Reyna", "Jett", "Deadlock"],
    ["Fracture", "Clove", "Chamber", "Sage", "Reyna", "Jett"],
    ["Fracture", "Clove", "Reyna", "Tejo", "Deadlock", "Phoenix"],
    ["Fracture", "Clove", "Sage", "Reyna", "Jett", "Deadlock"],
    ["Fracture", "Clove", "Chamber", "Reyna", "Neon", "Gekko"],
    ["Haven", "Clove", "Sage", "Breach", "Reyna", "Phoenix"],
    ["Haven", "Clove", "Sage", "Reyna", "Jett", "Deadlock"],
    ["Haven", "Cypher", "Sage", "Brimstone", "Reyna", "Jett"],
    ["Haven", "Cypher", "Clove", "Sage", "Reyna", "Neon"],
    ["Lotus", "Clove", "Breach", "Jett", "Deadlock", "Raze"],
    ["Lotus", "Iso", "Clove", "Killjoy", "Breach", "Jett"],
    ["Lotus", "Clove", "Chamber", "Reyna", "Neon", "Gekko"],
    ["Lotus", "Clove", "Killjoy", "Reyna", "Neon", "Fade"],
    ["Pearl", "Clove", "Jett", "Fade", "Phoenix", "Vyse"],
    ["Pearl", "Clove", "Killjoy", "Jett", "Neon", "Fade"],
    ["Pearl", "Iso", "Clove", "Killjoy", "Sova", "Jett"],
    ["Pearl", "Iso", "Cypher", "Clove", "Sova", "Jett"],
    ["Split", "Clove", "Sage", "Reyna", "Jett", "Deadlock"],
    ["Split", "Clove", "Sage", "Skye", "Jett", "Neon"],
    ["Split", "Clove", "Sage", "Jett", "Tejo", "Deadlock"],
    ["Split", "Cypher", "Clove", "Sage", "Jett", "Phoenix"]
]



agent_roles = {
    "Brimstone": "Controller", "Viper": "Controller", "Astra": "Controller", "Harbor": "Controller", "Clove": "Controller",
    "Phoenix": "Duelist", "Reyna": "Duelist", "Jett": "Duelist", "Raze": "Duelist", "Yoru": "Duelist", "Neon": "Duelist", "Iso": "Duelist",
    "Sage": "Sentinel", "Cypher": "Sentinel", "Killjoy": "Sentinel", "Chamber": "Sentinel", "Deadlock": "Sentinel",
    "Sova": "Initiator", "Breach": "Initiator", "Skye": "Initiator", "KAY/O": "Initiator", "Fade": "Initiator", "Gekko": "Initiator",
    "Vyse": "Sentinel", "Tejo": "Initiator", "Omen": "Controller"
}
def calculate_synergy(good_compositions, agents):
    synergy_scores = defaultdict(int)
    
    # Count agent appearances
    for comp in good_compositions:
        for agent in comp[1:]:  # Ignore the map name
            synergy_scores[agent] += 1
    
    # Normalize synergy scores (optional, can be skipped if not needed)
    max_score = max(synergy_scores.values(), default=1)
    for agent in synergy_scores:
        synergy_scores[agent] /= max_score
    
    # Assign 0 to agents not in good compositions
    for agent in agents:
        if agent not in synergy_scores:
            synergy_scores[agent] = 0
    
    return synergy_scores

# Run the function to get synergy scores
synergy_scores = calculate_synergy(good_compositions, agents)

# Print results
for agent, score in sorted(synergy_scores.items(), key=lambda x: -x[1]):
    print(f"{agent}: {score:.2f}")
synergy_scores = defaultdict(int)
for comp in good_compositions:
    for agent in comp[1:]:
        synergy_scores[agent] += 1
max_score = max(synergy_scores.values(), default=1)
for agent in synergy_scores:
    synergy_scores[agent] /= max_score
for agent in agents:
    if agent not in synergy_scores:
        synergy_scores[agent] = 0

def generate_team(map_name, good=True):
    valid_agents = set(agents)
    controllers = [a for a in valid_agents if agent_roles.get(a) == "Controller"]
    team = []
    role_counts = defaultdict(int)
    
    # Ensure a controller is in the team
    if good:
        controller = random.choice(controllers)
    else:
        controller = random.choice([a for a in valid_agents if a not in controllers])
    team.append(controller)
    role_counts[agent_roles.get(controller, "Unknown")] += 1
    valid_agents.remove(controller)
    
    while len(team) < 5:
        agent = (max(valid_agents, key=lambda a: synergy_scores[a] + random.random()) if good 
                else min(valid_agents, key=lambda a: synergy_scores[a] + random.random()))
        role = agent_roles.get(agent, "Unknown")
        
        if role_counts[role] < 2 and agent not in team:
            team.append(agent)
            role_counts[role] += 1
            valid_agents.remove(agent)
    
    return [map_name] + team

def generate_teams(good=True):
    map_names = set(comp[0] for comp in good_compositions)
    generated_teams = []
    for map_name in map_names:
        for _ in range(300):
            generated_teams.append(generate_team(map_name, good))
    return generated_teams

# Generate the final lists of compositions
dynamic_good_compositions = generate_teams(good=True)
dynamic_bad_compositions = generate_teams(good=False)
good_df = pd.DataFrame(dynamic_good_compositions)


good_df = pd.DataFrame(dynamic_good_compositions, columns=["Map", "Agent1", "Agent2", "Agent3", "Agent4", "Agent5"])

print(good_df)
good_df.to_csv('good_teams.csv', index=False)

