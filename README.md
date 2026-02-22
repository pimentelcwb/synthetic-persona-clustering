# synthetic-persona-clustering
LLM-powered pipeline to cluster structured data and synthesize UX personas.
# Synthetic Personas (Clustering + LLM)

This repo generates **one synthetic persona per cluster** from structured records:
- Create embeddings for each record
- Cluster similar records with KMeans
- Use a chat LLM to synthesize one persona per cluster
- Export results as JSON

## Quickstart

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
pip install -r requirements.txt
```

1) Create a `.env` file from `.env.example` and fill your endpoints/keys and paths.  
2) Run the persona pipeline:

```bash
python src/persona_pipeline.py
```

## Input format

Recommended input JSON:

```json
{
  "rows": [
    { "id": "1", "city": "Sao Paulo", "plan": "Premium", "main_pain": "..." },
    { "id": "2", "city": "Campinas", "plan": "Basic", "main_pain": "..." }
  ]
}
```

## Output

Writes a JSON to `OUTPUT_JSON_PATH` with clusters and one persona per cluster.

## Excel → JSON (optional)

If your source data is in Excel, use:

```bash
python src/excel_to_json.py
```

It converts `.xlsx` files into the `{"rows":[...]}` JSON format expected by the persona pipeline.

## Safety

- Do not commit `.env` or real datasets to GitHub.
- Keep real data local (e.g., in `data/`) and ignore it via `.gitignore`.
