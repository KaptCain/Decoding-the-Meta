# Decoding-the-Meta
 A Machine Learning Approach to Optimal Valorant Agent Compositions

Overview

This project consists of three main components designed to analyze and predict optimal agent selections and team compositions in the game Valorant. These tools are intended to help players and teams make data-driven decisions to improve their gameplay strategy. The tools include:

    Agent Fill Trainer: This tool predicts the best agent to add to a team based on the current map, team composition, and game rank.
    Best Agent Trainer: This tool identifies the best agent for a specific map and role, taking into account various gameplay statistics and rankings.
    Composition Generator and Trainer: This tool generates and evaluates team compositions, training a model to predict team performance based on rank, historical data and defined team synergy rules.

Project Structure

    bestAgentTrainer.py - Contains the Best Agent Trainer, which uses machine learning models to predict optimal agent choices based on the gameplay rank.
    compositionTrainer.py - Implements the Composition Model, which analyzes different team compositions to find the most effective combinations for given ranks and maps.
    agentFillTrainer.py - Powers the Agent Fill Trainer, recommending the best agent to complete a team's lineup effectively on a given map.

Each script is designed to be run independently, processing input data and producing predictions that can guide players in their game strategy development.
Installation

Before running the scripts, ensure that you have Python installed along with the following dependencies:

    pandas
    sklearn
    xgboost
    numpy

Data gathered from https://www.vstats.gg/

If you want to see how Vstats pull worked youll have to add your own chrome drivers to the file because the drivers are to large for me to upload with my repo.


Created by Alec Collins