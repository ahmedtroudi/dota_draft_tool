from fastapi import FastAPI,  HTTPException
from pydantic import BaseModel
from bayesian_model import recommend_hero, train_model
from preprocess_data import fetch_hero_mapping, OpenDotaCollector, preprocess_data
import arviz as az
import uvicorn

app = FastAPI()

class TeamComposition(BaseModel):
    allied: list[int]
    enemy: list[int]

@app.get("/data")
def retrieve_and_store_match_data():
    collector = OpenDotaCollector()
    collector.fetch_opendota_data(n_matches=50000)
    return {"message": "Hero Recommendation API is running!"}

@app.post("/train_model")
def train():
    collector = OpenDotaCollector()
    data = collector.fetch_opendota_data(n_matches=50000)
    processed_data = preprocess_data(data.head(500))
    model, trace = train_model(processed_data)
    az.to_netcdf(trace, "dota2_trace.nc")
    return {"message": "Train finished! Model trace saved!"}

@app.post("/recommend")
def recommend(team: TeamComposition):
    current_allied = team.allied
    current_enemy = team.enemy
    hero_mapping = fetch_hero_mapping()

    current_allied_names = [hero_mapping.get(ally, f"Unknown Hero {ally}") for ally in current_allied]
    current_enemy_names = [hero_mapping.get(enemy, f"Unknown Hero {enemy}") for enemy in current_enemy]

    if not current_allied or not current_enemy:
        raise HTTPException(status_code=400, detail="Both allied and enemy teams must have at least one hero.")

    trace = az.from_netcdf("dota2_trace.nc")
    recommendations = recommend_hero(current_allied, current_enemy, trace, hero_mapping)

    return {
        "allied_team": current_allied_names, # TODO: improve, explicitly provide if radiant or dire
        "enemy_team": current_enemy_names,
        "recommendations": [{"hero": hero_name, "win_probability": f"{win_prob:.2%}"}
                            for hero_name, win_prob in recommendations]
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, workers=1)