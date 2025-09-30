"""
plagiarism_detector.py
----------------------
Runtime plagiarism checker integrated with your MySQL schema.

Flow:
1. Load trained siamese model + threshold
2. Read all past submissions from Code_Submission (via db.py)
3. Compute embeddings (on demand from code_path)
4. Compare new submission vs past submissions
5. Update Code_Evaluation with plagiarism_score
6. Insert top matches into Plagiarism_match
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel, logging as hf_logging

from db import get_connection  # <-- your db.py connector

# Silence transformers logs
hf_logging.set_verbosity_error()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("plagiarism_detector")

# ----------------------------
# Config
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULTS = {
    "checkpoint": Path(
        os.getenv(
            "SIAMESE_CHECKPOINT",
            BASE_DIR / "siamese_model" / "siamese_plagiarism_best.pth",
        )
    ),
    "codebert_model": os.getenv("CODEBERT_MODEL", "microsoft/codebert-base"),
    "max_length": int(os.getenv("MAX_LENGTH", 512)),
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


# ----------------------------
# Siamese Model
# ----------------------------
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        return F.normalize(self.fc(x), p=2, dim=1)


# ----------------------------
# Embedder
# ----------------------------
class InferenceEmbedder:
    def __init__(self, model_name: str, device: torch.device, max_length: int):
        self.device = device
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = max_length

    def embed(self, code: str) -> torch.Tensor:
        if not code or not code.strip():
            return torch.zeros(self.model.config.hidden_size)
        with torch.no_grad():
            inputs = self.tokenizer(
                code, return_tensors="pt", truncation=True, max_length=self.max_length
            )
            for k in inputs:
                inputs[k] = inputs[k].to(self.device)
            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
        return emb


# ----------------------------
# DB Helpers
# ----------------------------
def load_submissions_from_db() -> Dict[str, str]:
    """
    Load past submissions from Code_Submission.
    Returns dict: submission_id -> code string.
    Resolves both relative and absolute paths.
    """
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT submission_id, code_path FROM Code_Submission")
    submissions = {}

    # Base directory where codes are stored
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "submitted_codes")
    )

    for row in cursor.fetchall():
        sid = str(row["submission_id"])
        path = row["code_path"]

        # Normalize path: if relative, make it absolute under submitted_codes
        if not os.path.isabs(path):
            path = os.path.join(
                base_dir, path.replace("submitted_codes/", "").replace("\\", "/")
            )

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                submissions[sid] = f.read()
        except FileNotFoundError:
            logger.warning("Skipping submission %s: file not found at %s", sid, path)
            continue
        except Exception as e:
            logger.warning(
                "Could not read file for submission %s (%s): %s", sid, path, e
            )
            continue

    cursor.close()
    conn.close()
    logger.info("Loaded %d submissions from DB", len(submissions))
    return submissions


def save_results(
    submission_id: int, plagiarism_score: float, matches: List[Tuple[str, float]]
):
    """
    Save plagiarism results into Code_Evaluation and Plagiarism_match.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Update Code_Evaluation with plagiarism_score
    cursor.execute(
        """
        UPDATE Code_Evaluation
        SET plagiarism_score = %s
        WHERE submission_id = %s
    """,
        (plagiarism_score, submission_id),
    )

    # Insert top matches into Plagiarism_match
    for mid, _ in matches:
        cursor.execute(
            """
            INSERT INTO Plagiarism_match (evaluation_id, matched_submission_id)
            VALUES (
                (SELECT code_evaluation_id FROM Code_Evaluation WHERE submission_id=%s),
                %s
            )
        """,
            (submission_id, mid),
        )

    conn.commit()
    cursor.close()
    conn.close()
    logger.info("Saved plagiarism results for submission %s", submission_id)


# ----------------------------
# Detector Loader
# ----------------------------
def load_detector(
    checkpoint_path: Path = DEFAULTS["checkpoint"],
    device: torch.device = DEFAULTS["device"],
):

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = SiameseNetwork()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    threshold = ckpt.get("threshold", 0.5)
    return model, threshold


# ----------------------------
# Embedding + Comparison
# ----------------------------
def compute_normalized_embedding(
    embedder: InferenceEmbedder, code: str, model: SiameseNetwork, device: torch.device
):
    print("inside compute normalized embeddig")
    raw_emb = embedder.embed(code).unsqueeze(0).to(device)
    with torch.no_grad():
        mapped = model(raw_emb).cpu().squeeze(0)
    return F.normalize(mapped, p=2, dim=0).numpy()


def compare(
    query_vec: np.ndarray, catalog: Dict[str, np.ndarray], top_k: int = 10
) -> List[Tuple[str, float]]:
    results = []
    q = query_vec.astype(np.float32)
    for key, vec in catalog.items():
        if isinstance(vec, torch.Tensor):
            vec = vec.numpy()
        v = vec / (np.linalg.norm(vec) + 1e-12)
        dist = np.linalg.norm(q - v)
        results.append((key, float(dist)))
    results.sort(key=lambda x: x[1])
    return results[:top_k]


# ----------------------------
# Main Detection
# ----------------------------
def check_plagiarism(
    submission_id: int, submission_code: str, top_k: int = 5
) -> Dict[str, Any]:

    model, threshold = load_detector()
    embedder = InferenceEmbedder(
        DEFAULTS["codebert_model"], DEFAULTS["device"], DEFAULTS["max_length"]
    )
    # Load all past submissions
    submissions = load_submissions_from_db()
    # Build catalog embeddings
    catalog = {}
    for sid, code in submissions.items():
        catalog[sid] = compute_normalized_embedding(
            embedder, code, model, DEFAULTS["device"]
        )

    # Compute query embedding
    query_vec = compute_normalized_embedding(
        embedder, submission_code, model, DEFAULTS["device"]
    )
    # Compare
    results = compare(
        query_vec, catalog, top_k + 1
    )  # fetch one extra in case of self-match

    # 🔥 Filter out self-matches (the current submission_id)
    results = [(sid, dist) for sid, dist in results if sid != str(submission_id)]

    # Keep only top_k after filtering
    results = results[:top_k]

    # Calculate plagiarism_score as 1 - min_distance/threshold (clipped 0-100)
    if results:
        best_dist = results[0][1]
        score = max(0.0, min(100.0, (1 - best_dist / threshold) * 100))
    else:
        score = 0.0
    # Save to DB
    save_results(submission_id, score, results)
    return {
        "submission_id": submission_id,
        "plagiarism_score": score,
        "threshold": threshold,
        "matches": [
            {
                "matched_submission_id": mid,
                "distance": dist,
                "is_plagiarism": dist < threshold,
            }
            for mid, dist in results
        ],
    }
