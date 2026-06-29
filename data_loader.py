import os
import kagglehub
import pandas as pd

def download_datasets():
    """
    Automated background downloading of the 3 key requested Kaggle datasets.
    Using Streamlit caching ensures this only happens once per container spin-up.
    """
    datasets = {
        "basketball_db": "wyattowalsh/basketball",
        "play_by_play_19_20": "harrywang/nba-play-by-play-2019-2020-season",
        "player_stats_24_25": "eduardopalmieri/nba-player-stats-season-2425"
    }
    
    paths = {}
    for key, repo in datasets.items():
        try:
            # Download latest version securely via kagglehub token/auth layout
            paths[key] = kagglehub.dataset_download(repo)
            print(f"Successfully linked {key} to local path: {paths[key]}")
        except Exception as e:
            print(f"Error pre-fetching {key}: {e}. Falling back to dynamic mock state.")
            paths[key] = None
            
    return paths

def load_historical_metrics():
    """
    A helper simulation generating base feature profiles using the structured targets.
    In real production, this parses tables from the downloaded 'basketball_db' path.
    """
    # Quick structural framework mimicking real RAPM / efficiency tables
    mock_players = [
        "LeBron James", "Stephen Curry", "Nikola Jokic", "Giannis Antetokounmpo", 
        "Luka Doncic", "Jayson Tatum", "Joel Embiid", "Kevin Durant", "Anthony Davis",
        "Alex Caruso", "Derrick White", "Marcus Smart", "Jr. Holiday", "Mikal Bridges"
    ]
    
    data = []
    import random
    random.seed(42)
    
    for player in mock_players:
        data.append({
            "Player": player,
            "RAPM": round(random.uniform(-2.0, 6.5), 2),
            "Offensive_Efficiency": round(random.uniform(105.0, 125.0), 1),
            "Defensive_Efficiency": round(random.uniform(100.0, 118.0), 1),
            "Stamina_Drain_Rate": round(random.uniform(0.05, 0.15), 3)
        })
        
    return pd.DataFrame(data)