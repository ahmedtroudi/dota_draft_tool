# dota_draft_tool

To run this app:

```
uvicorn app:app --host 0.0.0.0 --port 8083
```

Execute queries:

Train Model:
```
curl -X POST "http://localhost:8083/train_model"
```

Get recommendations:
```
❯ curl -X POST "http://localhost:8083/recommend" -H "Content-Type: application/json" -d '{"allied": [11, 13, 16], "enemy": [2, 7, 4]}'
```
