from collections import defaultdict
import random
import pandas as pd
import sqlLoader
from sqlLoader import SQLdb
from composition_generator import good_df as team_composition_df
from sklearn.metrics import accuracy_score
import numpy as np
from itertools import combinations
from sklearn.metrics import classification_report
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

import joblib
import pandas as pd
import numpy as np

# Load the pre-trained model and label encoders

import numpy as np
import pandas as pd
import joblib
from itertools import combinations

# Load pre-trained model and encoders
xgb_model = joblib.load('Resources/xgb_model.pkl')
label_encoders = joblib.load('Resources/label_encoders.pkl')
print("Available label encoders:", label_encoders.keys())

# Define valid maps and ranks
valid_maps = ["Abyss", "Bind", "Fracture", "Haven", "Lotus", "Pearl", "Split"]
valid_ranks = ["Iron", "Bronze", "Silver", "Gold", "Platinum", "Diamond", "Ascendant", "Immortal", "Radiant"]

# Encode map and rank dictionaries
map_dict = {name: i for i, name in enumerate(valid_maps)}
rank_dict = {name: i for i, name in enumerate(valid_ranks)}

# Get user input
while True:
    rank = input(f"Enter your rank {valid_ranks}: ").strip().capitalize()
    if rank in rank_dict:
        break
    print(f"Invalid rank '{rank}'. Please choose from {valid_ranks}")

while True:
    map_name = input(f"Enter the map name {valid_maps}: ").strip().capitalize()
    if map_name in map_dict:
        break
    print(f"Invalid map '{map_name}'. Please choose from {valid_maps}")

# Encode map and rank
encoded_map = map_dict[map_name]
encoded_rank = rank_dict[rank]
df = pd.read_csv('Resources/team_composition_with_winrates3.csv')  # Ensure this file exists
unique_agents = set()

# Collect unique agent names from all agent columns
def get_best_team(rank, map_name, xgb_model, label_encoders, team_df):
    """ Predicts the best team composition for a given rank and map using preset teams. """

    # Filter dataset for the given map and rank
    filtered_df = team_df[(team_df["Map"] == map_name) & (team_df["Rank"] == rank)]

    if filtered_df.empty:
        print(f"No valid teams found for {rank} on {map_name}.")
        return None, None

    best_team = None
    best_winrate = 0

    # Iterate through predefined teams
    for _, row in filtered_df.iterrows():
        team = [row["Agent1"], row["Agent2"], row["Agent3"], row["Agent4"], row["Agent5"]]

        # Encode selected team agents
        encoded_agents = [label_encoders["agent"].transform([agent])[0] if agent in label_encoders["agent"].classes_ else -1 for agent in team]

        # One-hot encode the rank
        rank_columns = ['Rank_Ascendant', 'Rank_Bronze', 'Rank_Diamond', 'Rank_Gold', 'Rank_Immortal',
                        'Rank_Iron', 'Rank_Platinum', 'Rank_Radiant', 'Rank_Silver']
        rank_encoding = [1 if f"Rank_{rank}" in rank_columns else 0 for rank in rank_columns]

        # One-hot encode the map
        map_columns = ['Map_0', 'Map_1', 'Map_2', 'Map_3', 'Map_4', 'Map_5', 'Map_6']
        map_index = valid_maps.index(map_name)  # Find the correct map index
        map_encoding = [1 if i == map_index else 0 for i in range(len(map_columns))]

        # Feature placeholders for missing role columns (Duelist, Sentinel, Controller, Initiator)
        role_encoding = [0, 0, 0, 0]  # Adjust if roles are known

        # Ensure 'Valid_Comp' column is included (set to 1 for testing)
        valid_comp = [1]

        # Construct the full input feature vector
        input_data = np.array([encoded_agents + role_encoding + rank_encoding + map_encoding + valid_comp])

        # Predict win rate
        predicted_winrate = xgb_model.predict(input_data)[0]

        # Keep the best team
        if predicted_winrate > best_winrate:
            best_winrate = predicted_winrate
            best_team = team

    return best_team, best_winrate

best_team, predicted_winrate = get_best_team(rank, map_name, xgb_model, label_encoders, df)

# Display the best team composition
print(f" **Best Team Composition for {rank} on {map_name}:**")
print("Agents:", best_team)
print(f"Predicted Win Rate: {predicted_winrate:.2f}")

