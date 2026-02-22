import os
import json
from typing import Any, Dict, List, Optional, cast

import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =========================
# CONFIG
# =========================

# ---- Paths
INPUT_JSON = os.getenv("INPUT_JSON_PATH", r"your_path/input_file.json")
OUTPUT_JSON = os.getenv("OUTPUT_JSON_PATH", r"your_path/output_file.json")

# ---- Embeddings
EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT", "your_url")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "your_api_key")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT", "your_embedding_deployment_name")

# ---- Chat
CHAT_ENDPOINT = os.getenv("CHAT_ENDPOINT", "your_url")
CHAT_API_KEY = os.getenv("CHAT_API_KEY", "your_api_key")
CHAT_DEPLOYMENT = os.getenv("CHAT_DEPLOYMENT", "your_chat_deployment_name")

# ---- Clustering
K_MIN = int(os.getenv("K_MIN", "3"))
K_MAX = int(os.getenv("K_MAX", "8"))

# If set, overrides automatic K choice (e.g., "10")
FORCE_K_RAW = os.getenv("FORCE_K", "").strip()
FORCE_K = int(FORCE_K_RAW) if FORCE_K_RAW.isdigit() else None

# ---- Embedding batching
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

# ---- LLM sampling per cluster
MAX_ROWS_CLUSTER = int(os.getenv("MAX_ROWS_CLUSTER", "20"))


# =========================
# CLIENTS
# =========================

embedding_client = OpenAI(
    base_url=EMBEDDING_ENDPOINT,
    api_key=EMBEDDING_API_KEY,
)

chat_client = OpenAI(
    base_url=CHAT_ENDPOINT,
    api_key=CHAT_API_KEY,
)


# =========================
# HELPERS
# =========================

def load_rows() -> List[Dict[str, Any]]:
    """Load rows from INPUT_JSON.

    Supports:
      - {"rows": [...]} (dict with "rows")
      - [...] (list)
    """
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        j = json.load(f)

    if isinstance(j, dict) and "rows" in j and isinstance(j["rows"], list):
        return j["rows"]
    if isinstance(j, list):
        return j

    raise ValueError("Unexpected JSON format. Expected {'rows':[...]} or a plain list [...].")


def row_to_text(row: Dict[str, Any]) -> str:
    """Convert one JSON record into a text string for embeddings.

    To reduce noise and cost, it ignores NA/empty values.
    """
    parts: List[str] = []
    for k in sorted(row.keys()):
        v = row.get(k, "NA")
        if v is None:
            v = "NA"
        v = str(v).strip()
        if v == "" or v.upper() == "NA":
            continue
        parts.append(f"{k}: {v}")
    return " | ".join(parts)


def embed_texts(texts: List[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    """Create embeddings for a list of texts using batching."""
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = embedding_client.embeddings.create(
            model=EMBEDDING_DEPLOYMENT,
            input=batch,
        )
        vectors.extend([x.embedding for x in resp.data])
        print(f"Embedding {i + len(batch)}/{len(texts)}")

    return np.array(vectors, dtype=np.float32)


def choose_k(X: np.ndarray) -> int:
    """Choose K using silhouette score over the configured [K_MIN..K_MAX] range."""
    if len(X) < (K_MIN + 2):
        return 2

    best_k = K_MIN
    best_score = -1.0

    upper = min(K_MAX, len(X) - 1)
    for k in range(K_MIN, upper + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)

        # Silhouette may fail if any cluster ends up with 1 item.
        try:
            score = silhouette_score(X, labels)
        except Exception:
            continue

        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def compact_row_for_llm(row: Dict[str, Any], max_fields: int = 80) -> Dict[str, Any]:
    """Reduce a record before sending to the LLM.

    - keeps "id" if present
    - includes only fields != NA
    - limits number of fields to control prompt size
    """
    out: Dict[str, Any] = {}
    if "id" in row:
        out["id"] = str(row["id"])

    count = 0
    for k in sorted(row.keys()):
        if k == "id":
            continue
        v = row.get(k, "NA")
        if v is None:
            v = "NA"
        v = str(v).strip()
        if v == "" or v.upper() == "NA":
            continue
        out[k] = v
        count += 1
        if count >= max_fields:
            break

    return out


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Try to extract the first JSON object {...} from a text blob. Returns dict or None."""
    if not text:
        return None

    # If it already looks like pure JSON
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def synthesize_persona(cluster_id: int, cluster_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Use the chat model to synthesize one persona JSON for a cluster."""
    sample = cluster_rows[:MAX_ROWS_CLUSTER]
    sample_compact = [compact_row_for_llm(row) for row in sample]

    schema_hint = {
        "persona_id": f"persona_{cluster_id}",
        "title": "",
        "summary": "",
        "traits": [],
        "needs": [],
        "pain_points": [],
        "motivations": [],
        "behaviors": [],
        "values": [],
        "living_situation": [],
        "children_profile": [],
        "support_signals": [],
        "hypotheses": [],
        "representative_ids": [],
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior UX/CX researcher. "
                "Generate ONE synthetic persona based on the provided examples. "
                "Do not invent attributes that are not supported by the data; if you infer anything, put it under 'hypotheses'. "
                "Return ONLY valid JSON (no extra text)."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"cluster_id": cluster_id, "schema": schema_hint, "sample": sample_compact},
                ensure_ascii=False,
            ),
        },
    ]

    persona: Optional[Dict[str, Any]] = None
    last_content = ""

    # 1) Try JSON mode (if supported)
    try:
        resp = chat_client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            temperature=0.3,
            messages=messages, # type: ignore
            response_format={"type": "json_object"},
        )
        last_content = resp.choices[0].message.content or ""
        persona = json.loads(last_content)

    except Exception:
        # 2) Fallback without response_format
        resp = chat_client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            temperature=0.3,
            messages=messages, # type: ignore
        )
        last_content = (resp.choices[0].message.content or "").strip()
        persona = _extract_json_object(last_content)

        # 3) One-shot "fix JSON" attempt
        if persona is None:
            fix = chat_client.chat.completions.create(
                model=CHAT_DEPLOYMENT,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "Return ONLY valid JSON. No extra text."},
                    {"role": "user", "content": last_content},
                ],
            )
            fixed = (fix.choices[0].message.content or "").strip()
            persona = _extract_json_object(fixed)

            if persona is None:
                raise ValueError(f"LLM did not return valid JSON. Content: {fixed[:300]}")

    # At this point persona must be a dict
    assert persona is not None

    # Ensure required fields exist
    if not persona.get("representative_ids"):
        persona["representative_ids"] = [
            str(row.get("id", "")) for row in sample if row.get("id") is not None
        ][:10]

    if not persona.get("persona_id"):
        persona["persona_id"] = f"persona_{cluster_id}"

    return persona


# =========================
# MAIN
# =========================

def main() -> None:
    rows = load_rows()
    if not rows:
        raise ValueError("Input JSON has no rows.")

    texts = [row_to_text(r) for r in rows]
    embeddings = embed_texts(texts)

    # K selection
    k = FORCE_K if FORCE_K is not None else choose_k(embeddings)

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(embeddings)

    # Group records by cluster
    clusters: Dict[int, List[Dict[str, Any]]] = {}
    for i, c in enumerate(labels):
        clusters.setdefault(int(c), []).append(rows[i])

    personas_out: List[Dict[str, Any]] = []
    # Largest clusters first
    for cid, items in sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True):
        print(f"Generating persona for cluster {cid} (n={len(items)}) ...")
        persona = synthesize_persona(cid, items)

        personas_out.append(
            {
                "cluster_id": cid,
                "cluster_size": len(items),
                "persona": persona,
            }
        )

    output = {
        "input_file": os.path.basename(INPUT_JSON),
        "total_records": len(rows),
        "clusters": k,
        "personas": personas_out,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("✔ Done. Output file:", OUTPUT_JSON)


if __name__ == "__main__":
    main()