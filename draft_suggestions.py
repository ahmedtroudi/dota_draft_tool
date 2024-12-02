import argparse
from preprocess_data import fetch_hero_mapping, OpenDotaCollector, preprocess_data
from bayesian_model import recommend_hero, train_model

def main():
    parser = argparse.ArgumentParser(description="Hero Recommendation Tool")
    parser.add_argument(
        "--allied",
        nargs="+",
        type=int,
        required=True,
        help="List of hero IDs for allied team (e.g., 29 138 110)"
    )
    parser.add_argument(
        "--enemy",
        nargs="+",
        type=int,
        required=True,
        help="List of hero IDs for enemy team (e.g., 23 63 26)"
    )
    args = parser.parse_args()
    collector = OpenDotaCollector()
    data = collector.fetch_opendota_data(n_matches=10000)
    processed_data = preprocess_data(data)
    model, trace = train_model(processed_data)

    hero_mapping = fetch_hero_mapping()

    current_allied_names = [hero_mapping.get(ally, f"Unknown Hero {ally}") for ally in args.allied]
    current_enemy_names = [hero_mapping.get(enemy, f"Unknown Hero {enemy}") for enemy in args.enemy]

    print(f"Allied team consists of: {current_allied_names}")
    print(f"Enemy team consists of: {current_enemy_names}")

    recommendations = recommend_hero(args.allied, args.enemy, trace, hero_mapping)
    for hero_name, win_prob in recommendations:
        print(f"Hero: {hero_name}, Win Probability: {win_prob:.2%}")

if __name__ == "__main__":
    main()