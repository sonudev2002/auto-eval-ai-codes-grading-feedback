"""
training_model.py
-----------------
Robust training module for siamese plagiarism detector.

Drop into:
D:\mca_final_project\backend\siamese_model\training_model.py

Features:
- Preprocess Java files (clean comments / normalize whitespace).
- Build pairs dataset (positive / negative / cross-case negatives).
- Precompute CodeBERT embeddings (batched, cached).
- Siamese model with contrastive loss and 5-fold CV training.
- Threshold tuning and checkpoint saving (model + threshold).
- CLI entrypoints for: preprocess, build_pairs, embed, train, tune
- Defensive checks, logging, GPU handling, reproducibility.
"""

import os
import re
import math
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import random

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import RobertaTokenizer, RobertaModel, logging as hf_logging

# Reduce transformers verbosity
hf_logging.set_verbosity_error()

# ----------------------------
# Configuration (edit as needed)
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
# Logging and seeding
# ----------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("siamese_training")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(DEFAULTS["seed"])


# ----------------------------
# Utilities
# ----------------------------
def safe_mkdir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def clean_java_code(code: str) -> str:
    if not isinstance(code, str):
        return ""
    # Remove single-line comments //
    code = re.sub(r"//.*", "", code)
    # Remove multi-line comments /* ... */
    code = re.sub(r"/\*[\s\S]*?\*/", "", code)
    # Normalize whitespace
    code = re.sub(r"\s+", " ", code)
    return code.strip()


# ----------------------------
# Preprocessing & Pair Generation
# ----------------------------
def preprocess_dataset(root: Path, out_csv: Path) -> pd.DataFrame:
    """
    Walk root/ and extract Java files from 'original', 'plagiarized', 'non-plagiarized' subfolders.
    Saves CSV with columns: case, category, filename (relative), code
    """
    logger.info("Preprocessing dataset from %s -> %s", root, out_csv)
    records = []
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    for case in sorted([p.name for p in root.iterdir() if p.is_dir()]):
        case_path = root / case
        for category in ("original", "plagiarized", "non-plagiarized"):
            cat_path = case_path / category
            if not cat_path.exists():
                continue
            for dirpath, _, filenames in os.walk(cat_path):
                for fname in filenames:
                    if not fname.lower().endswith(".java"):
                        continue
                    fpath = Path(dirpath) / fname
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                            raw = fh.read()
                        cleaned = clean_java_code(raw)
                        if cleaned:
                            rel = str(fpath.relative_to(root))
                            records.append(
                                {
                                    "case": case,
                                    "category": category,
                                    "filename": rel,
                                    "code": cleaned,
                                }
                            )
                    except Exception as exc:
                        logger.warning("Skipping %s due to %s", fpath, exc)

    df = pd.DataFrame(records)
    safe_mkdir(out_csv)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info("Saved preprocessed CSV: %s (%d rows)", out_csv, len(df))
    return df


def build_pairs(preprocessed_csv: Path, out_pairs_csv: Path) -> pd.DataFrame:
    """
    Build pairs (file1, file2, label) from preprocessed CSV.
    label: 1 -> plagiarized / same (positive), 0 -> not plagiarized (negative)
    """
    logger.info("Building pairs from %s -> %s", preprocessed_csv, out_pairs_csv)
    df = pd.read_csv(preprocessed_csv)
    pairs = []
    cases = df.groupby("case")

    for case, group in cases:
        originals = group[group["category"] == "original"]
        plagiarized = group[group["category"] == "plagiarized"]
        nonplag = group[group["category"] == "non-plagiarized"]

        # original vs plagiarized
        if not originals.empty and not plagiarized.empty:
            orig_file = originals.iloc[0]["filename"]
            for _, r in plagiarized.iterrows():
                pairs.append((orig_file, r["filename"], 1))

        # plag vs plag (intra-case)
        plag_files = plagiarized["filename"].tolist()
        for i in range(len(plag_files)):
            for j in range(i + 1, len(plag_files)):
                pairs.append((plag_files[i], plag_files[j], 1))

        # original vs nonplag
        if not originals.empty and not nonplag.empty:
            orig_file = originals.iloc[0]["filename"]
            for _, r in nonplag.iterrows():
                pairs.append((orig_file, r["filename"], 0))

        # plag vs nonplag
        for _, plag in plagiarized.iterrows():
            for _, np_ in nonplag.iterrows():
                pairs.append((plag["filename"], np_["filename"], 0))

    # Cross-case hard negatives: pair non-plag files from different cases (sampled)
    all_nonplag = df[df["category"] == "non-plagiarized"]["filename"].tolist()
    random.shuffle(all_nonplag)
    for i in range(0, len(all_nonplag) - 1, 2):
        pairs.append((all_nonplag[i], all_nonplag[i + 1], 0))

    pairs_df = pd.DataFrame(pairs, columns=["file1", "file2", "label"])
    safe_mkdir(out_pairs_csv)
    pairs_df.to_csv(out_pairs_csv, index=False)
    logger.info("Saved pairs CSV: %s (%d rows)", out_pairs_csv, len(pairs_df))
    return pairs_df


# ----------------------------
# Embedding with CodeBERT
# ----------------------------
class Embedder:
    def __init__(
        self,
        model_name: str = DEFAULTS["codebert_model"],
        device: Optional[torch.device] = None,
        max_length: int = DEFAULTS["max_length"],
    ):
        self.device = (
            device
            if device is not None
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = max_length

    def get_embedding(self, code: str) -> torch.Tensor:
        if not code or not isinstance(code, str):
            # return zero vector to avoid crashing downstream
            emb_size = self.model.config.hidden_size
            return torch.zeros(emb_size)
        with torch.no_grad():
            inputs = self.tokenizer(
                code, return_tensors="pt", truncation=True, max_length=self.max_length
            )
            # move to device
            for k in list(inputs.keys()):
                inputs[k] = inputs[k].to(self.device)
            outputs = self.model(**inputs)
            # CLS token embedding
            emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
        return emb

    def batch_embeddings(self, code_list: List[Tuple[str, str]], batch_size: int = 8):
        """
        code_list: list of tuples (filename, code)
        yields (filename, embedding)
        """
        # Simple batching: tokenize per item to avoid huge tensors
        for i in range(0, len(code_list), batch_size):
            batch = code_list[i : i + batch_size]
            texts = [c for (_, c) in batch]
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            for k in list(inputs.keys()):
                inputs[k] = inputs[k].to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_embs = outputs.last_hidden_state[:, 0, :].cpu()
            for (fname, _), emb in zip(batch, cls_embs):
                yield fname, emb


def precompute_embeddings(
    preprocessed_csv: Path,
    embed_out: Path,
    batch_size: int = DEFAULTS["batch_size_embed"],
    model_name: str = DEFAULTS["codebert_model"],
):
    """
    Compute and save embeddings dict {filename: tensor}
    """
    logger.info("Precomputing embeddings from %s -> %s", preprocessed_csv, embed_out)
    df = pd.read_csv(preprocessed_csv)
    embedder = Embedder(model_name=model_name)
    records = []
    # Build code list
    items = [(row["filename"], row["code"]) for _, row in df.iterrows()]
    embeddings: Dict[str, torch.Tensor] = {}
    for fname, emb in tqdm(
        embedder.batch_embeddings(items, batch_size=batch_size),
        total=math.ceil(len(items) / batch_size),
    ):
        embeddings[fname] = emb
    safe_mkdir(embed_out)
    # Save in CPU tensors
    cpu_map = {k: v.cpu() for k, v in embeddings.items()}
    torch.save(cpu_map, embed_out)
    logger.info("Saved %d embeddings to %s", len(cpu_map), embed_out)
    return cpu_map


# ----------------------------
# Dataset & Model
# ----------------------------
class CodePairsDataset(Dataset):
    def __init__(self, pairs_df: pd.DataFrame, embeddings: Dict[str, torch.Tensor]):
        self.df = pairs_df.reset_index(drop=True)
        self.embeddings = embeddings

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        f1, f2, label = row["file1"], row["file2"], float(row["label"])
        emb1 = self.embeddings.get(f1)
        emb2 = self.embeddings.get(f2)
        if emb1 is None or emb2 is None:
            # Return zeros if missing (but log once)
            logger.debug("Missing embedding for %s or %s", f1, f2)
            emb_size = next(iter(self.embeddings.values())).shape[0]
            emb1 = torch.zeros(emb_size)
            emb2 = torch.zeros(emb_size)
        return emb1, emb2, torch.tensor(label, dtype=torch.float32)


def collate_pairs(batch):
    emb1 = torch.stack(
        [b[0] if isinstance(b[0], torch.Tensor) else torch.tensor(b[0]) for b in batch]
    )
    emb2 = torch.stack(
        [b[1] if isinstance(b[1], torch.Tensor) else torch.tensor(b[1]) for b in batch]
    )
    labels = torch.stack([b[2] for b in batch])
    return emb1.float(), emb2.float(), labels.float()


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

    def forward(self, x1, x2):
        out1 = self.fc(x1)
        out2 = self.fc(x2)
        out1 = F.normalize(out1, p=2, dim=1)
        out2 = F.normalize(out2, p=2, dim=1)
        return out1, out2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        distances = F.pairwise_distance(out1, out2)
        loss_pos = label * torch.pow(distances, 2)
        loss_neg = (1 - label) * torch.pow(
            torch.clamp(self.margin - distances, min=0.0), 2
        )
        return torch.mean(loss_pos + loss_neg)


# ----------------------------
# Training loop + CV
# ----------------------------
def train_fold(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-4,
    embedding_dim: int = 768,
):
    model = SiameseNetwork(embedding_dim=embedding_dim).to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for emb1, emb2, labels in train_loader:
            emb1 = emb1.to(device)
            emb2 = emb2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out1, out2 = model(emb1, emb2)
            loss = criterion(out1, out2, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train = total_train_loss / max(1, len(train_loader))

        # validation
        model.eval()
        total_val_loss = 0.0
        preds = []
        trues = []
        with torch.no_grad():
            for emb1, emb2, labels in val_loader:
                emb1 = emb1.to(device)
                emb2 = emb2.to(device)
                labels = labels.to(device)
                out1, out2 = model(emb1, emb2)
                loss = criterion(out1, out2, labels)
                total_val_loss += loss.item()
                dist = F.pairwise_distance(out1, out2)
                pred = (dist < 0.5).float()
                preds.extend(pred.cpu().numpy())
                trues.extend(labels.cpu().numpy())

        avg_val = total_val_loss / max(1, len(val_loader))
        acc = accuracy_score(trues, preds) if trues else 0.0
        f1 = f1_score(trues, preds, zero_division=0) if trues else 0.0
        logger.info(
            "Epoch %d | TrainLoss %.4f | ValLoss %.4f | Acc %.3f | F1 %.3f",
            epoch + 1,
            avg_train,
            avg_val,
            acc,
            f1,
        )

        scheduler.step(avg_val)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict().copy()

    return best_state, best_val_loss


def run_crossval(
    pairs_csv: Path,
    embeddings_pt: Path,
    out_model: Path,
    n_splits: int = DEFAULTS["n_splits"],
    epochs: int = 10,
    device: Optional[torch.device] = None,
    batch_size: int = DEFAULTS["batch_size_train"],
):
    """
    Run K-Fold cross validation, save best model by validation loss to out_model.
    """
    device = (
        device
        if device is not None
        else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    )
    logger.info("Starting cross-validation on device %s", device)

    pairs_df = pd.read_csv(pairs_csv)
    embeddings = torch.load(embeddings_pt, map_location="cpu")
    dataset = CodePairsDataset(pairs_df, embeddings)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=DEFAULTS["seed"])
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):
        logger.info("Fold %d/%d", fold + 1, n_splits)
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_pairs,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_pairs,
            pin_memory=True,
        )

        best_state, best_val_loss = train_fold(
            train_loader, val_loader, device, epochs=epochs
        )
        fold_results.append((best_state, best_val_loss))
        logger.info("Fold %d done. Best val loss: %.4f", fold + 1, best_val_loss)

    # choose best overall
    best_state, best_loss = min(fold_results, key=lambda x: x[1])
    # Save model state (non-threshold)
    safe_mkdir(out_model)
    torch.save(best_state, out_model)
    logger.info("Saved best model state to %s (val loss %.4f)", out_model, best_loss)
    return out_model


# ----------------------------
# Final test + threshold tuning
# ----------------------------
def tune_threshold(
    pairs_csv: Path,
    embeddings_pt: Path,
    model_checkpoint: Path,
    out_checkpoint: Path,
    test_size: float = 0.15,
    device: Optional[torch.device] = None,
):
    device = (
        device
        if device is not None
        else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    )
    logger.info("Tuning threshold using model %s", model_checkpoint)
    pairs_df = pd.read_csv(pairs_csv)
    embeddings = torch.load(embeddings_pt, map_location="cpu")

    train_val_df, test_df = train_test_split(
        pairs_df,
        test_size=test_size,
        random_state=DEFAULTS["seed"],
        stratify=pairs_df["label"],
    )
    test_dataset = CodePairsDataset(test_df.reset_index(drop=True), embeddings)
    test_loader = DataLoader(
        test_dataset, batch_size=DEFAULTS["batch_size_train"], collate_fn=collate_pairs
    )

    model = SiameseNetwork().to(device)
    # model_checkpoint is state_dict
    state = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    distances = []
    trues = []
    with torch.no_grad():
        for emb1, emb2, labels in test_loader:
            emb1 = emb1.to(device)
            emb2 = emb2.to(device)
            out1, out2 = model(emb1, emb2)
            dist = F.pairwise_distance(out1, out2).cpu().numpy()
            distances.extend(dist.tolist())
            trues.extend(labels.numpy().tolist())

    distances = np.array(distances)
    trues = np.array(trues).astype(int)

    # sweep thresholds
    best_f1 = -1
    best_t = None
    best_prec = best_rec = 0.0
    for t in np.linspace(0.1, 1.5, 80):
        preds = (distances < t).astype(int)
        f1 = f1_score(trues, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            best_prec = precision_score(trues, preds, zero_division=0)
            best_rec = recall_score(trues, preds, zero_division=0)

    logger.info(
        "Best threshold: %.4f | F1: %.4f | Prec: %.4f | Rec: %.4f",
        best_t,
        best_f1,
        best_prec,
        best_rec,
    )

    # Save checkpoint with model state + tuned threshold and meta
    safe_mkdir(out_checkpoint)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "threshold": float(best_t),
        "meta": {
            "best_f1": float(best_f1),
            "precision": float(best_prec),
            "recall": float(best_rec),
        },
    }
    torch.save(checkpoint, out_checkpoint)
    logger.info("Saved tuned checkpoint to %s", out_checkpoint)
    return out_checkpoint


# ----------------------------
# CLI helpers
# ----------------------------
def ensure_files_exist(*paths: Path):
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")


# Example usage functions (you can call from CLI or import)
def full_pipeline(data_root: Optional[str] = None):
    cfg = DEFAULTS.copy()
    if data_root:
        cfg["data_root"] = Path(data_root)

    preprocessed = preprocess_dataset(
        Path(cfg["data_root"]), Path(cfg["preprocessed_csv"])
    )
    pairs = build_pairs(Path(cfg["preprocessed_csv"]), Path(cfg["pairs_csv"]))
    precompute_embeddings(
        Path(cfg["preprocessed_csv"]),
        Path(cfg["embeddings_pt"]),
        batch_size=cfg["batch_size_embed"],
        model_name=cfg["codebert_model"],
    )
    run_crossval(
        Path(cfg["pairs_csv"]),
        Path(cfg["embeddings_pt"]),
        Path(cfg["model_out"]),
        n_splits=cfg["n_splits"],
        epochs=10,
    )
    tune_threshold(
        Path(cfg["pairs_csv"]),
        Path(cfg["embeddings_pt"]),
        Path(cfg["model_out"]),
        Path(cfg["best_out"]),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Siamese plagiarism training utilities"
    )
    parser.add_argument(
        "--action",
        choices=["preprocess", "pairs", "embed", "train", "tune", "pipeline"],
        required=True,
    )
    parser.add_argument("--data_root", default=str(DEFAULTS["data_root"]))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        if args.action == "preprocess":
            preprocess_dataset(Path(args.data_root), DEFAULTS["preprocessed_csv"])
        elif args.action == "pairs":
            build_pairs(DEFAULTS["preprocessed_csv"], DEFAULTS["pairs_csv"])
        elif args.action == "embed":
            precompute_embeddings(
                DEFAULTS["preprocessed_csv"],
                DEFAULTS["embeddings_pt"],
                batch_size=DEFAULTS["batch_size_embed"],
            )
        elif args.action == "train":
            run_crossval(
                DEFAULTS["pairs_csv"],
                DEFAULTS["embeddings_pt"],
                DEFAULTS["model_out"],
                epochs=args.epochs,
                device=device,
            )
        elif args.action == "tune":
            tune_threshold(
                DEFAULTS["pairs_csv"],
                DEFAULTS["embeddings_pt"],
                DEFAULTS["model_out"],
                DEFAULTS["best_out"],
            )
        elif args.action == "pipeline":
            full_pipeline(args.data_root)
    except Exception as e:
        logger.exception("Operation failed: %s", e)
        raise
