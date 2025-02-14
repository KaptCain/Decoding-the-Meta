import sqlLoader
import pandas as pd
from sqlLoader import SQLdb
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from collections import Counter
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
# Load full dataset    
    
df = SQLdb()

df_all_ranks = df[df["rank"] != "ALL"].copy()

# Label Encoding for categorical columns
map_encoder = LabelEncoder()
role_encoder = LabelEncoder()
agent_encoder = LabelEncoder()
rank_encoder = LabelEncoder()

df_all_ranks["map_encoded"] = map_encoder.fit_transform(df_all_ranks["map"])
df_all_ranks["role_encoded"] = role_encoder.fit_transform(df_all_ranks["role"])
df_all_ranks["agent_encoded"] = agent_encoder.fit_transform(df_all_ranks["agent"])
df_all_ranks["rank_encoded"] = rank_encoder.fit_transform(df_all_ranks["rank"])

# Compute final weighted win rate
max_matches = df_all_ranks["totalmatches"].max()
df_all_ranks["weighted_winrate"] = (
    df_all_ranks["winrate"] * df_all_ranks["totalmatches"]
) / max_matches

# Compute map-specific agent win rate
df_all_ranks["map_agent_winrate"] = df_all_ranks.groupby(["map", "agent"])['winrate'].transform('mean')

df_all_ranks["map_importance"] = df_all_ranks["map_encoded"] * 3  # Increase map influence

# Feature selection
X_task1 = df_all_ranks[[
    "map_encoded", "role_encoded", "rank_encoded",
    "winrate", "weighted_winrate", "map_agent_winrate", "map_importance"
]]

y_task1 = df_all_ranks["agent_encoded"]

# Use MinMaxScaler instead of StandardScaler
scaler = MinMaxScaler()
X_task1_scaled = scaler.fit_transform(X_task1)

# Train-test split


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(X_task1_scaled, y_task1):
    X_train_task1, X_test_task1 = X_task1_scaled[train_index], X_task1_scaled[test_index]
    y_train_task1, y_test_task1 = y_task1.iloc[train_index], y_task1.iloc[test_index]


xgb_task1 = xgb.XGBClassifier(
    eval_metric="mlogloss",
    n_estimators=100,  # Reduce number of trees
    learning_rate=0.03,  # Lower learning rate
    max_depth=3,  # Reduce depth to prevent memorization
    subsample=0.7,  # Reduce the percentage of data per tree
    colsample_bytree=0.7,  # Use fewer features per tree
    reg_alpha=0.3,  # Increase L1 regularization (forces model to be simpler)
    reg_lambda=0.5  # Increase L2 regularization (penalizes complexity)
)


# Perform cross-validation
cv_scores = cross_val_score(xgb_task1, X_task1_scaled, y_task1, cv=5, scoring="accuracy")

# Print cross-validation results
print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")
xgb_task1.fit(X_train_task1, y_train_task1)

# Make predictions
# Make predictions on both training and validation sets
y_train_pred_task1 = xgb_task1.predict(X_train_task1)
y_test_pred_task1 = xgb_task1.predict(X_test_task1)

# Calculate training and validation accuracy
train_accuracy = accuracy_score(y_train_task1, y_train_pred_task1)
test_accuracy = accuracy_score(y_test_task1, y_test_pred_task1)

# Print both accuracies to check for overfitting
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Validation Accuracy: {test_accuracy:.2f}")

# Print classification report for detailed evaluation
print("\nClassification Report (Validation Set):")
print(classification_report(y_test_task1, y_test_pred_task1))


# Calculate accuracy
accuracy_task1 = accuracy_score(y_test_task1, y_test_pred_task1)
print("XGBoost Accuracy:", accuracy_task1)

# Get user input for map, role, and rank
map_name = input("Enter the map name (Abyss, Bind, Fracture, Haven, Lotus, Pearl, Split): ").strip().capitalize()
role_name = input("Enter the role (Duelist, Controller, Sentinel, Initiator): ").strip().capitalize()
rank_name = input("Enter your rank (Iron, Bronze, Silver, Gold, Platinum, Diamond, Ascendant, Immortal, Radiant): ").strip().capitalize()

# Validate input
if map_name not in df_all_ranks["map"].unique():
    print(f"Error: {map_name} is not a valid map. Please check your spelling.")
    exit()

if role_name not in df_all_ranks["role"].unique():
    print(f"Error: {role_name} is not a valid role. Choose from Duelist, Controller, Sentinel, or Initiator.")
    exit()

if rank_name not in df_all_ranks["rank"].unique():
    print(f"Error: {rank_name} is not a valid rank. Please check your spelling.")
    exit()

# Encode user input
map_encoded = map_encoder.transform([map_name])[0] * 3  # Apply the new map weight
role_encoded = role_encoder.transform([role_name])[0]
rank_encoded = rank_encoder.transform([rank_name])[0]

# Compute role-based win rate from dataset
map_role_winrate = df_all_ranks[(df_all_ranks["map"] == map_name) & (df_all_ranks["role"] == role_name) & (df_all_ranks["rank"] == rank_name)]["winrate"].mean()
map_role_winrate = map_role_winrate if not np.isnan(map_role_winrate) else df_all_ranks["winrate"].mean()

# Construct input array
input_data = np.array([[map_encoded, role_encoded, rank_encoded,
                        df_all_ranks["winrate"].mean(), df_all_ranks["weighted_winrate"].mean(),
                        map_role_winrate, map_encoded]])

input_df = pd.DataFrame(input_data, columns=X_task1.columns)
input_data_scaled = scaler.transform(input_df)

# Get predictions for all agents
all_predictions = xgb_task1.predict_proba(input_data_scaled)

# Convert back to agent names
predicted_agents = agent_encoder.inverse_transform(np.argsort(all_predictions[0])[::-1])

# Filter only agents that match the requested role
valid_agents = df_all_ranks[df_all_ranks["role"] == role_name]["agent"].unique()

# Find the first predicted agent that matches the role
best_agent = next((agent for agent in predicted_agents if agent in valid_agents), "No valid agent found")

print(f" Best Agent for {map_name} as {role_name} at {rank_name}: {best_agent}")
    
