import requests
import pandas as pd
import numpy as np
import os
import logging
import time
import ast


class OpenDotaCollector:
    def __init__(self, rate_limit_per_min: int = 1200, api_key="insert-your-key-here"): # not needed if you have the csv files
        # TODO: DO NOT COMMIT API_KEY
        self.base_url = "https://api.opendota.com/api/publicMatches"
        self.api_key = api_key
        self.rate_limit = rate_limit_per_min
        self.last_request_time = 0

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _rate_limit_wait(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        wait_time = (60 / self.rate_limit) - time_since_last

        if wait_time > 0:
            time.sleep(wait_time)

        self.last_request_time = time.time()

    def fetch_opendota_data(self, n_matches=10000, file_path="opendota_data.csv"):
        if os.path.exists(file_path):
            self.logger.info(f"Loading data from {file_path}")
            return pd.read_csv(file_path)
        else:
            self.logger.info(f"Fetching data from OpenDota API")
            all_matches = []
            last_match_id = None

            while len(all_matches) < n_matches:
                self._rate_limit_wait()

                params = {
                    "less_than_match_id": last_match_id,
                    "api_key": self.api_key,
                }
                response = requests.get(self.base_url, params=params)

                if response.status_code == 200:
                    data = response.json()
                    if not data:
                        self.logger.info("No more data available from API.")
                        break

                    all_matches.extend(data)
                    last_match_id = data[-1]['match_id']
                    self.logger.info(f"Fetched {len(data)} matches. Total: {len(all_matches)}")
                else:
                    raise ValueError(f"Error fetching data from OpenDota API: {response.status_code}")

            df = pd.DataFrame(all_matches[:n_matches])
            df.to_csv(file_path, index=False)
            return df

def fetch_hero_mapping(file_path="hero_mapping.csv"):
    if os.path.exists(file_path):
        print(f"Loading hero mapping from {file_path}")
        df = pd.read_csv(file_path)
        return dict(zip(df['id'], df['localized_name']))
    else:
        url = "https://api.opendota.com/api/heroes"
        response = requests.get(url)
        if response.status_code == 200:
            hero_data = response.json()
            hero_mapping = {hero['id']: hero['localized_name'] for hero in hero_data}
            df = pd.DataFrame(hero_data)
            df.to_csv(file_path, index=False)
            return hero_mapping
        else:
            raise ValueError("Error fetching hero data from OpenDota API")

def fetch_historical_hero_win_rates(file_path="hero_win_rates.csv"):
    if os.path.exists(file_path):
        print(f"Loading hero win rates from {file_path}")
        df = pd.read_csv(file_path)
        return df['win_rate'].to_numpy()
    else:
        url = "https://api.opendota.com/api/heroStats"
        try:
            response = requests.get(url)
            response.raise_for_status()
            hero_stats = response.json()

            total_heroes = 118  # TODO: improve and use logging
            hero_win_rates = np.full(total_heroes, 0.5)

            csv_data = []

            for hero in hero_stats:
                hero_id = hero.get("id", None)
                if hero_id is not None and 1 <= hero_id <= total_heroes:
                    try:
                        win_rate = hero["pro_win"] / hero["pro_pick"] if hero["pro_pick"] > 0 else 0.5
                    except Exception:  # TODO: improve
                        win_rate = 0.5
                    hero_win_rates[hero_id - 1] = win_rate
                    csv_data.append({"id": hero_id, "win_rate": win_rate})

            pd.DataFrame(csv_data).to_csv(file_path, index=False)
            return hero_win_rates

        except requests.RequestException as e:
            print(f"Error fetching data from OpenDota API: {e}")
            return np.full(118, 0.5)


def preprocess_data(data):
    def extract_teams(row):
        radiant_team = [int(hero) for hero in ast.literal_eval(row['radiant_team'])]
        dire_team = [int(hero) for hero in ast.literal_eval(row['dire_team'])]
        return radiant_team, dire_team

    data['radiant_team'], data['dire_team'] = zip(*data.apply(extract_teams, axis=1))
    data['outcome'] = data['radiant_win'].astype(int)  # radiant win as outcome (1 for win, 0 for loss)
    return data[['radiant_team', 'dire_team', 'outcome']]



