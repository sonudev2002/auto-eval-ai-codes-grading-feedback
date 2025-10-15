"""
plagiarism_detector.py
----------------------
Runtime plagiarism checker integrated with a MySQL schema.

Flow:
1. Load trained Siamese model + threshold
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
from backend.db import get_connection
import requests

# Silence transformers logs and configure logger
hf_logging.set_verbosity_error()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("plagiarism_detector")

# ----------------------------
# Configuration
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
# Siamese Model Definition
# ----------------------------
class SiameseNetwork(nn.Module):
    """Fully connected projection head for embedding normalization."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing L2-normalized embeddings."""
        return F.normalize(self.fc(x), p=2, dim=1)


# ----------------------------
# Embedding Utility
# ----------------------------
class InferenceEmbedder:
    """Encodes source code into embeddings using CodeBERT."""

    def __init__(self, model_name: str, device: torch.device, max_length: int):
        self.device = device
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.max_length = max_length

    def embed(self, code: str) -> torch.Tensor:
        """Generate a fixed-size embedding for the given code snippet."""
        if not code or not code.strip():
            return torch.zeros(self.model.config.hidden_size)
        with torch.no_grad():
            inputs = self.tokenizer(
                code, return_tensors="pt", truncation=True, max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
        return emb


# ----------------------------
# Database Utilities
# ----------------------------
def load_submissions_from_db() -> Dict[str, str]:
    """
    Load past submissions from Code_Submission table.

    Returns:
        dict: {submission_id: code_string}
    Supports both local file paths and remote (HTTP) sources.
    """
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT submission_id, code_path FROM code_submission")
    submissions = {}

    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "submitted_codes")
    )

    for row in cursor.fetchall():
        sid = str(row["submission_id"])
        path = (row.get("code_path") or "").strip()

        if not path:
            logger.warning("Skipping submission %s: empty code_path", sid)
            continue

        # Handle remote URLs
        if path.startswith(("http://", "https://")):
            try:
                resp = requests.get(path, timeout=10)
                if resp.ok:
                    submissions[sid] = resp.text
                else:
                    logger.warning("HTTP %s fetching %s", resp.status_code, path)
            except Exception as e:
                logger.warning("Failed to fetch %s (%s): %s", sid, path, e)
            continue

        # Handle local paths
        if not os.path.isabs(path):
            path = os.path.join(
                base_dir, path.replace("submitted_codes/", "").replace("\\", "/")
            )

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                submissions[sid] = f.read()
        except FileNotFoundError:
            logger.warning("File not found for submission %s at %s", sid, path)
        except Exception as e:
            logger.warning("Error reading %s: %s", sid, e)

    cursor.close()
    conn.close()
    logger.info("Loaded %d submissions from DB", len(submissions))
    return submissions


def save_results(
    submission_id: int, plagiarism_score: float, matches: List[Tuple[str, float]]
):
    """Persist plagiarism results into Code_Evaluation and Plagiarism_match."""
    conn = get_connection()
    cursor = conn.cursor()

    # Update evaluation score
    cursor.execute(
        """
        UPDATE code_evaluation
        SET plagiarism_score = %s
        WHERE submission_id = %s
        """,
        (plagiarism_score, submission_id),
    )

    # Record top matches
    for mid, _ in matches:
        cursor.execute(
            """
            INSERT INTO plagiarism_match (evaluation_id, matched_submission_id)
            VALUES (
                (SELECT code_evaluation_id FROM code_evaluation WHERE submission_id=%s),
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
# Model Loading
# ----------------------------
def load_detector(
    checkpoint_path: Path = DEFAULTS["checkpoint"],
    device: torch.device = DEFAULTS["device"],
) -> Tuple[SiameseNetwork, float]:
    """Load Siamese model and its similarity threshold from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SiameseNetwork()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    threshold = ckpt.get("threshold", 0.5)
    return model, threshold


# ----------------------------
# Embedding & Comparison
# ----------------------------
def compute_normalized_embedding(
    embedder: InferenceEmbedder, code: str, model: SiameseNetwork, device: torch.device
) -> np.ndarray:
    """Compute normalized embedding vector for a code snippet."""
    raw_emb = embedder.embed(code).unsqueeze(0).to(device)
    with torch.no_grad():
        mapped = model(raw_emb).cpu().squeeze(0)
    return F.normalize(mapped, p=2, dim=0).numpy()


def compare(
    query_vec: np.ndarray, catalog: Dict[str, np.ndarray], top_k: int = 10
) -> List[Tuple[str, float]]:
    """Compute L2 distance between query and catalog embeddings."""
    q = query_vec.astype(np.float32)
    results = [
        (key, float(np.linalg.norm(q - (vec / (np.linalg.norm(vec) + 1e-12)))))
        for key, vec in catalog.items()
    ]
    results.sort(key=lambda x: x[1])
    return results[:top_k]


# ----------------------------
# Main Detection Logic
# ----------------------------
def check_plagiarism(
    submission_id: int, submission_code: str, top_k: int = 5
) -> Dict[str, Any]:
    """Detect plagiarism for a given submission and update results in DB."""
    model, threshold = load_detector()
    embedder = InferenceEmbedder(
        DEFAULTS["codebert_model"], DEFAULTS["device"], DEFAULTS["max_length"]
    )

    # Load all past submissions
    submissions = load_submissions_from_db()

    # Build embeddings for the catalog
    catalog = {
        sid: compute_normalized_embedding(embedder, code, model, DEFAULTS["device"])
        for sid, code in submissions.items()
    }

    # Compute query embedding
    query_vec = compute_normalized_embedding(
        embedder, submission_code, model, DEFAULTS["device"]
    )

    # Find closest matches
    results = compare(query_vec, catalog, top_k + 1)
    results = [(sid, dist) for sid, dist in results if sid != str(submission_id)][
        :top_k
    ]

    # Compute plagiarism score (scaled 0–100)
    score = 0.0
    if results:
        best_dist = results[0][1]
        score = max(0.0, min(100.0, (1 - best_dist / threshold) * 100))

    # Persist results
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
