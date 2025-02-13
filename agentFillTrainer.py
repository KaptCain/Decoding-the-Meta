import sqlLoader
import pandas as pd
from sqlLoader import SQLdb
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost
from sklearn.metrics import accuracy_score
import numpy as np
from itertools import combinations
from sklearn.metrics import classification_report, confusion_matrix

# Load full dataset
df = SQLdb()

df_all_ranks = df[df["rank"] != "ALL"].copy()

# Label Encoding for categorical columns
map_encoder = LabelEncoder()
agent_encoder = LabelEncoder()
role_encoder = LabelEncoder()
rank_encoder = LabelEncoder()
map_encoder.fit(df_all_ranks["map"].unique().tolist() + ["Abyss", "Bind", "Fracture", "Haven", "Lotus", "Pearl", "Split"])

rank_encoder.fit(df_all_ranks["rank"].unique().tolist() + ["Iron", "Bronze", "Silver", "Gold", "Platinum", "Diamond", "Ascendant", "Immortal", "Radiant"])



df_all_ranks["map_encoded"] = map_encoder.fit_transform(df_all_ranks["map"])
df_all_ranks["agent_encoded"] = agent_encoder.fit_transform(df_all_ranks["agent"])
df_all_ranks["role_encoded"] = role_encoder.fit_transform(df_all_ranks["role"])
df_all_ranks["rank_encoded"] = rank_encoder.fit_transform(df_all_ranks["rank"])
# Compute team synergy score based on past compositions
df_all_ranks["synergy_score"] = df_all_ranks.groupby(["map", "agent"])['winrate'].transform('mean')

# Good team compositions
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

# Generate training data from good team compositions
X_train = []
y_train = []

while True:
            rank_name = input("Enter your rank (Iron, Bronze, Silver, Gold, Platinum, Diamond, Ascendant, Immortal, Radiant): ").strip().capitalize()
            if rank_name in rank_encoder.classes_:
                break
            print(f"Invalid rank name '{rank_name}'. Available ranks: {rank_encoder.classes_}")
for comp in good_compositions:
    map_name = comp[0]
    agents = comp[1:]
    for combo in combinations(agents, 4):
        if map_name not in map_encoder.classes_:
            raise ValueError(f"Map '{map_name}' is not in the encoder! Available maps: {map_encoder.classes_}")
        
            
  
        encoded_rank = rank_encoder.transform([rank_name])[0]

        encoded_map = map_encoder.transform([map_name])[0]
        
        encoded_agents = [agent_encoder.transform([a])[0] for a in combo]
        target_agent = list(set(agents) - set(combo))[0]
        X_train.append([encoded_map] + encoded_agents)
        y_train.append(agent_encoder.transform([target_agent])[0])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Fix label encoding issue
y_train_encoder = LabelEncoder()
y_train = y_train_encoder.fit_transform(y_train)

# Scale data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
xgb_model = xgboost.XGBClassifier(
    eval_metric="mlogloss",
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.6,
    reg_alpha=0.1,
    reg_lambda=0.1
)
xgb_model.fit(X_train_scaled, y_train)

# Evaluate model
accuracy = accuracy_score(y_train, xgb_model.predict(X_train_scaled))
print(f"Model Accuracy: {accuracy:.2f}")

# Define function to predict best agent
def recommend_best_agent(map_name, selected_agents, rank_name):
    if len(selected_agents) >= 5:
        return "Team is already full!"
    
    # Get roles for already selected agents
    selected_roles = df_all_ranks[df_all_ranks['agent'].isin(selected_agents)][['agent', 'role']]
    role_counts = selected_roles['role'].value_counts().to_dict()
    
    encoded_map = map_encoder.transform([map_name])[0]
    encoded_agents = [agent_encoder.transform([a])[0] for a in selected_agents]
    while len(encoded_agents) < 4:
        encoded_agents.append(-1)  # Placeholder
    
    input_data = np.array([[encoded_map] + encoded_agents])
    input_scaled = scaler.transform(input_data)
    
    
  
   


    # Predict agent
    all_predictions = xgb_model.predict_proba(input_scaled)[0]  # Get prediction probabilities
    sorted_agents = np.argsort(all_predictions)[::-1]  # Sort by probability

# Find the highest-ranked agent that is NOT already selected
    best_agent = None
    for agent_idx in sorted_agents:
        candidate_agent = agent_encoder.inverse_transform([agent_idx])[0]
        if candidate_agent not in selected_agents:
            best_agent = candidate_agent
            break  # Stop at the first valid agent

# If all agents are already selected (highly unlikely), return an error message
    if best_agent is None:
        return "Error: No available agents to recommend."

# Ensure role constraints (max 2 per role)
    best_agent_role = df_all_ranks[df_all_ranks['agent'] == best_agent]['role'].values[0]
    if role_counts.get(best_agent_role, 0) >= 2:
        for agent_idx in sorted_agents:
            potential_agent = agent_encoder.inverse_transform([agent_idx])[0]
            potential_agent_role = df_all_ranks[df_all_ranks['agent'] == potential_agent]['role'].values[0]
            if potential_agent not in selected_agents and role_counts.get(potential_agent_role, 0) < 2:
                best_agent = potential_agent
                break

        return f" Best agent to pick for {map_name}, in rank {rank_name} with {', '.join(selected_agents)}: {best_agent}"

    if len(selected_agents) >= 5:
        return "Team is already full!"
    
    encoded_map = map_encoder.transform([map_name])[0]
    encoded_agents = [agent_encoder.transform([a])[0] for a in selected_agents]
    while len(encoded_agents) < 4:
        encoded_agents.append(-1)  # Placeholder
    
    input_data = np.array([[encoded_map] + encoded_agents])
    input_scaled = scaler.transform(input_data)
    
    predicted_agent = xgb_model.predict(input_scaled)[0]
    best_agent = agent_encoder.inverse_transform([predicted_agent])[0]
    
    return f" Best agent to pick for {map_name}, in rank {rank_name} with {', '.join(selected_agents)}: {best_agent}"

# Interactive input for map and selected agents
# Interactive input for map and rank
while True:
    map_name = input("Enter the map name (Abyss, Bind, Fracture, Haven, Lotus, Pearl, Split): ").strip().capitalize()
    if map_name in map_encoder.classes_:
        break
    print("Invalid map name. Please enter a valid map.(Abyss, Bind, Fracture, Haven, Lotus, Pearl, Split)") 



# Interactive input for selected agents
selected_agents = []
print("Enter selected agents one by one. Type 'next' when done:")
while True:
    agent = input("Agent: ").strip().capitalize()
    if agent.lower() == 'next':
        break
    if agent in df_all_ranks["agent"].unique():
        selected_agents.append(agent)
    else:
        print("Invalid agent name. Try again.")



# Model Performance Summary
print("\n **Model Performance Summary** \n")

# Accuracy
print(f" Model Accuracy: {accuracy:.4f}")  # Display accuracy

# Classification Report
y_train_pred = xgb_model.predict(X_train_scaled)
print("\n Classification Report:\n", classification_report(y_train, y_train_pred, zero_division=1))

# Confusion Matrix
conf_matrix = confusion_matrix(y_train, y_train_pred)
print("\n Confusion Matrix:\n", conf_matrix)

# Feature Importance (Which factors influence predictions the most?)
feature_importance = xgb_model.feature_importances_
feature_names = ["map_encoded"] + [f"Agent{i+1}_encoded" for i in range(4)]

print("\n Feature Importances:")
for name, importance in zip(feature_names, feature_importance):
    print(f"  - {name}: {importance:.4f}")

# Model Hyperparameters
print("\n Model Hyperparameters:")
print(xgb_model.get_params())


print(recommend_best_agent(map_name, selected_agents, rank_name))