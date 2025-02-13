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
# List of all Valorant agents

agent_winrate_df = SQLdb()
print(agent_winrate_df)
# Get unique ranks from the agent winrate dataset
unique_ranks = agent_winrate_df["rank"].unique()
team_composition_df = pd.read_csv('Resources/good_teams.csv')
# Assuming the structure of your DataFrame is consistent with the indices you've provided
team_composition_df.columns = [ 'Map', 'Agent1', 'Agent2', 'Agent3', 'Agent4', 'Agent5']

  # Adjust path as necessary
# Create a new dataframe to store all ranks for each team composition
expanded_team_composition = []
# print(team_composition_df)
# # Iterate over each team composition and create copies for each rank
# for index, row in team_composition_df.iterrows():
#     for rank in unique_ranks:
#         team_data = row.copy()
#         team_data["Rank"] = rank  # Assign rank dynamically
#         expanded_team_composition.append(team_data)

# # Convert the expanded list to a DataFrame
# expanded_team_composition_df = pd.DataFrame(expanded_team_composition)
# print(expanded_team_composition_df)
# def calculate_synergy(agents, agent_winrate_df):
#     role_counts = {"Duelist": 0, "Sentinel": 0, "Controller": 0, "Initiator": 0}
    
#     for agent in agents:
#         role = agent_winrate_df[agent_winrate_df['agent'] == agent]['role'].values[0]
#         role_counts[role] += 1
    
#     # Define some synergy rules (you can adjust these based on the actual gameplay data you have)
#     synergy_score = 0
#     if role_counts["Controller"] >= 1:
#         synergy_score += 0.2  # Controllers add synergy
#     if role_counts["Duelist"] <= 2:
#         synergy_score += 0.02  # Duelists are better in smaller numbers
#     if role_counts["Sentinel"] <= 2:
#         synergy_score += 0.05  # Sentinels should be limited
#     if role_counts["Initiator"] >= 1:
#         synergy_score += 0.2  # Initiators help with better control
    
#     return synergy_score
# # Function to get the correct win rate for each agent based on Map and Rank
# def get_winrate(map_name, rank, agent, agent_winrate_df):
#     row = agent_winrate_df[(agent_winrate_df["map"] == map_name) & 
#                         (agent_winrate_df["rank"] == rank) & 
#                         (agent_winrate_df["agent"] == agent)]
#     return row["winrate"].values[0] if not row.empty else 0

# # Compute the average team win rate for each map-rank combination including synergy
# avg_win_rates = []
# for index, row in expanded_team_composition_df.iterrows():
#     map_name = row["Map"]
#     rank = row["Rank"]
#     agents = [row["Agent1"], row["Agent2"], row["Agent3"], row["Agent4"], row["Agent5"]]
    
#     # Get individual agent win rates for the specified map and rank
#     win_rates = [get_winrate(map_name, rank, agent, agent_winrate_df) for agent in agents]
    
#     # Compute the team synergy score
#     synergy_score = calculate_synergy(agents, agent_winrate_df)
    
#     # Compute the average win rate and adjust by synergy score
#     avg_win_rate = (sum(win_rates) / len(win_rates)) + synergy_score
#     avg_win_rates.append(avg_win_rate)

# # Add the computed average win rate to the dataset
# expanded_team_composition_df["Avg_Winrate"] = avg_win_rates

# # Save the updated DataFrame to a CSV file
# expanded_team_composition_df.to_csv('Resources/team_composition_with_winrates3.csv', index=False)

#Display the correctly updated dataset



#Add the computed average win rate to the dataset


######################################################################################################
######################This code above takes a minute to load but just combines some data sets
######################i didnt want to run it everytime so i saved it in a csv that i loaded from there
#######################################################################################################
#Display updated dataset




team_composition_df = pd.read_csv('Resources/team_composition_with_winrates3.csv', index_col=False)
print(team_composition_df)







label_encoders = {}

# Encode the "Map" column
label_encoders["Map"] = LabelEncoder()
team_composition_df["Map"] = label_encoders["Map"].fit_transform(team_composition_df["Map"])

# Encode agents
for agent_col in ["Agent1", "Agent2", "Agent3", "Agent4", "Agent5"]:
    label_encoders[agent_col] = LabelEncoder()
    team_composition_df[agent_col] = label_encoders[agent_col].fit_transform(team_composition_df[agent_col])
# Load agent win rate dataset to get agent roles
agent_roles = agent_winrate_df[["agent", "role"]].drop_duplicates()

# Melt team composition dataset to long format (one row per agent per team)
team_composition_long = team_composition_df.melt(
    id_vars=["Map", "Avg_Winrate"],
    value_vars=["Agent1", "Agent2", "Agent3", "Agent4", "Agent5"],
    var_name="AgentPosition",
    value_name="Agent"
)
team_composition_long["Agent"] = team_composition_long["Agent"].astype(str)
agent_roles["agent"] = agent_roles["agent"].astype(str)

# Merge agent roles to get the role of each agent in the team
team_composition_long = team_composition_long.merge(
    agent_roles, left_on="Agent", right_on="agent", how="left"
).drop(columns=["agent"])

# Count how many of each role exist per team
role_counts = team_composition_long.groupby(["Map", "Avg_Winrate"]).role.value_counts().unstack().fillna(0)

# Ensure all role columns exist (add missing ones as 0)
for role in ["Duelist", "Sentinel", "Controller", "Initiator"]:
    if role not in role_counts.columns:
        role_counts[role] = 0  # Add missing role column

# Merge role features back into original dataset
team_composition_df = team_composition_df.merge(role_counts, on=["Map", "Avg_Winrate"], how="left").fillna(0)

# One-hot encode "Rank" and "Map"
team_composition_df = pd.get_dummies(team_composition_df, columns=["Rank", "Map"])

# Define a function to check valid compositions
def is_valid_composition(row):
    return (row["Controller"] >= 1) and (row["Duelist"] <= 2) and (row["Sentinel"] <= 2) and (row["Initiator"] <= 2)

# Apply the function to create a new feature
team_composition_df["Valid_Comp"] = team_composition_df.apply(is_valid_composition, axis=1).astype(int)  # 1 if valid, 0 if not

# Define features (X) and target variable (y)
X = team_composition_df.drop(columns=["Avg_Winrate"])  # Remove win rate from features
y = team_composition_df["Avg_Winrate"]  # Use win rate as target



# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training Data Columns:", X_train.columns)



# Ensure "Avg_Winrate" is removed from training and testing data
if "Avg_Winrate" in X_train.columns:
    X_train = X_train.drop(columns=["Avg_Winrate"])
    X_test = X_test.drop(columns=["Avg_Winrate"])






# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],  # Reduced upper limit
    'learning_rate': [0.01, 0.02, 0.03],  # Lower learning rates
    'max_depth': [3, 5],  # Reduced depth
    'reg_alpha': [0.3, 0.5, 0.7],  # Increased L1 regularization
    'reg_lambda': [1.0, 1.2, 1.5]  # Increased L2 regularization
}

# Initialize XGBoost model
xgb_regressor = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

# Run Grid Search
grid_search = GridSearchCV(xgb_regressor, param_grid, cv=3, scoring="r2", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get best parameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Train model with best parameters
xgb_regressor = xgb.XGBRegressor(**best_params)
xgb_regressor.fit(X_train, y_train)

# Make predictions
y_pred = xgb_regressor.predict(X_test)


# Evaluate model performance
rmse = mean_squared_error(y_test, y_pred, squared=False)  
r2 = r2_score(y_test, y_pred)  


#  Model Performance Summary
print("\n **Model Performance Summary** \n")
print(f" Optimized RMSE: {rmse:.4f}")  # Lower is better
print(f" Optimized R² Score: {r2:.4f}")  # Closer to 1 is better
print(f" Best Model Parameters: {best_params}\n")


# Save the model to a file
joblib.dump(xgb_regressor, 'Resources/xgb_model.pkl')
# Save LabelEncoders after training
joblib.dump(label_encoders, 'Resources/label_encoders.pkl')

