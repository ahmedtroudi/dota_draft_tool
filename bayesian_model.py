import pymc as pm
import numpy as np
from preprocess_data import fetch_historical_hero_win_rates
N_HEROES = 118


def build_model(_data):
    with pm.Model() as _model:
        # Historical win rates for prior
        historical_win_rates = fetch_historical_hero_win_rates()
        log_odds_win_rates = np.log(historical_win_rates / (1 - historical_win_rates))

        hero_strength = pm.Normal("hero_strength", mu=log_odds_win_rates, sigma=0.5, shape=N_HEROES)

        counter_effects = pm.Normal("counter_effects", mu=0, sigma=0.8, shape=(N_HEROES, N_HEROES))

        radiant_matrix = np.array([
            [1 if hero in team else 0 for hero in range(N_HEROES)]
            for team in _data['radiant_team']
        ])
        dire_matrix = np.array([
            [1 if hero in team else 0 for hero in range(N_HEROES)]
            for team in _data['dire_team']
        ])

        cross_team_interactions = pm.math.dot(radiant_matrix, pm.math.dot(counter_effects, dire_matrix.T)).diagonal()

        radiant_strength = pm.math.dot(radiant_matrix, hero_strength) + cross_team_interactions
        dire_strength = pm.math.dot(dire_matrix, hero_strength)
        delta_strength = radiant_strength - dire_strength

        # Noise as epsilon
        # epsilon = pm.HalfNormal("epsilon", sigma=0.05)
        p_win = pm.Deterministic("p_win", pm.math.sigmoid(delta_strength))
        observed = pm.Bernoulli("observed", p=p_win, observed=_data['outcome'])

    return _model

def train_model(_data):
    _model = build_model(_data)
    with _model:
        _trace = pm.sample(draws=5000, tune=1000, target_accept=0.95, return_inferencedata=True, cores=8)
    return _model, _trace


def recommend_hero(current_allied, current_enemy, trace, hero_mapping):
    """
    Recommends heroes based on the current picks and the trace of the trained model.
    """
    hero_strength = trace.posterior.hero_strength.mean(axis=(0, 1))
    counter_effects = trace.posterior.counter_effects.mean(axis=(0, 1))
    n_heroes = hero_strength.shape[0]

    allied_vector = np.zeros(n_heroes)
    enemy_vector = np.zeros(n_heroes)

    # Mark currently picked heroes
    for hero in current_allied:
        allied_vector[hero] = 1
    for hero in current_enemy:
        enemy_vector[hero] = 1

    win_probs = []
    for hero in range(n_heroes):
        if hero not in current_allied + current_enemy:  # Skip already-picked heroes
            test_allied_vector = allied_vector.copy()
            test_allied_vector[hero] = 1  # Simulate picking this hero

            cross_interaction = np.dot(test_allied_vector, np.dot(counter_effects, enemy_vector.T))

            radiant_strength = np.dot(test_allied_vector, hero_strength) + cross_interaction
            dire_strength = np.dot(enemy_vector, hero_strength)
            delta_strength = radiant_strength - dire_strength

            win_prob = 1 / (1 + np.exp(-delta_strength))  # sigmoid
            win_probs.append((hero, win_prob))

    # sort recommendations by win probability (descending order)
    top_recommendations = sorted(win_probs, key=lambda x: x[1], reverse=True)[:5]

    return [(hero_mapping.get(hero_id, f"Unknown Hero {hero_id}"), prob) for hero_id, prob in top_recommendations]

