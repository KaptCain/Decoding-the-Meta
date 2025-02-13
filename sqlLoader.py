import psycopg2
import pandas as pd
from sqlalchemy import create_engine
def SQLdb():
    DB_NAME = "valorant_agent_stats_db"
    DB_USER = "postgres"
    DB_PASS = "postgres"
    DB_HOST = "localhost"  # or your host
    DB_PORT = "5432"

# Connect to PostgreSQL
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

    query = """
    SELECT map, rank, agent, role, winrate, totalmatches
    FROM valorant_agent_stats
    WHERE Rank in ( 'Radiant', 'Immortal', 'Ascendant', 'Diamond', 'Platinum', 'Gold', 'Silver', 'Bronze', 'Iron' );
    """

# Load data into a Pandas DataFrame
    df = pd.read_sql(query, engine)

    from sklearn.preprocessing import MinMaxScaler

# Normalize the Win Rate (0 to 1)
    scaler = MinMaxScaler()
    df["winrate"] = scaler.fit_transform(df[["winrate"]])

    return df
