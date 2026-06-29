import numpy as np

class NBAPredictionModel:
    """
    An algorithmic blueprint tracking structural components requested by the role specification.
    Takes complex real-time User Inputs and maps them to structural Model Outputs.
    """
    def __init__(self):
        pass

    def evaluate_lineup_metric(self, lineup_data):
        """Calculates combined base strengths of the 5-player floor lineup."""
        avg_rapm = lineup_data["RAPM"].mean()
        avg_off = lineup_data["Offensive_Efficiency"].mean()
        avg_def = lineup_data["Defensive_Efficiency"].mean()
        return avg_rapm, avg_off, avg_def

    def predict(self, game_context, home_lineup, away_lineup):
        """
        Processes multi-dimensional inputs to return live game actionable inferences.
        """
        # Extract context inputs
        score_diff = game_context["home_score"] - game_context["away_score"]
        sec_left = (game_context["minutes_left"] * 60) + game_context["seconds_left"]
        possession = game_context["possession"]
        
        # Extract player impacts
        h_rapm, h_off, h_def = self.evaluate_lineup_metric(home_lineup)
        a_rapm, a_off, a_def = self.evaluate_lineup_metric(away_lineup)
        
        # Calculate Base Win Probability via logistic sigmoid mapping 
        # Accounts for time decaying variance (leverage gets higher as time winds down)
        time_factor = max(sec_left, 1) / 2880  # Normalized total game length
        leverage = 1.0 / (time_factor + 0.1)
        
        # Score margin baseline + lineup power differences
        z_score = (score_diff * 0.25 * leverage) + ((h_rapm - a_rapm) * 0.4)
        if possession == "Home":
            z_score += 0.15
        else:
            z_score -= 0.15
            
        win_prob = 1 / (1 + np.exp(-z_score))
        win_prob = np.clip(win_prob, 0.01, 0.99)
        
        # Calculate dynamic Lineup Efficiency Forecasts based on context & fatigue
        # Assumes fatigue marginally degrades offensive production over a long stint
        fatigue_penalty = (5 - game_context["recent_turnovers"] * 0.2)
        predicted_off_rating = h_off - fatigue_penalty
        predicted_def_rating = h_def + (game_context["recent_fouls"] * 0.5)
        
        # Return bundled structural outputs
        return {
            "win_probability": float(win_prob),
            "forecasted_offensive_rating": round(predicted_off_rating, 1),
            "forecasted_defensive_rating": round(predicted_def_rating, 1)
        }