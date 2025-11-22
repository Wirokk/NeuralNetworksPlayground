import os
import json
import pickle
from typing import Dict, Any, List, Optional

from neural_network import Network  # pour les types / compat éventuelle

# Dossier racine où tout est sauvegardé
RUNS_DIR = "runs"
HISTORY_PATH = os.path.join(RUNS_DIR, "runs_history.json")


# ============================================================
# Utils de base
# ============================================================

def _ensure_dirs() -> None:
    """Crée le dossier runs/ si besoin."""
    os.makedirs(RUNS_DIR, exist_ok=True)


def _run_dir(run_id: str) -> str:
    return os.path.join(RUNS_DIR, run_id)


# ============================================================
# History JSON (pour le Hall of Fame)
# ============================================================

def load_runs_history() -> List[Dict[str, Any]]:
    """Charge la liste des runs depuis runs_history.json."""
    _ensure_dirs()
    if not os.path.exists(HISTORY_PATH):
        return []

    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        # En cas de fichier corrompu, on repart de zéro
        return []


def save_runs_history(history: List[Dict[str, Any]]) -> None:
    """Écrit la liste des runs dans runs_history.json."""
    _ensure_dirs()
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def get_hof_top3(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Retourne les 3 meilleurs runs triés par accuracy décroissante."""
    if not history:
        return []
    sorted_hist = sorted(
        history,
        key=lambda x: x.get("final_accuracy", 0.0),
        reverse=True,
    )
    return sorted_hist[:3]


# ============================================================
# Sauvegarde d’un run complet
# ============================================================

def save_full_run(
    run_id: str,
    net: Network,
    final_accuracy: float,
    metrics_history: List[Dict[str, Any]],
    weight_history: Dict[str, Any],
    misclassified: List[Any],
    config: Dict[str, Any],
) -> None:
    """
    Sauvegarde *tout* ce qui concerne un run dans runs/<run_id>/ :

      - model.pkl        : dict {sizes, weights, biases}
      - metrics.pkl      : liste de dicts par epoch
      - weights.pkl      : historique des poids (première couche)
      - misclassified.pkl: exemples mal classés
      - config.pkl       : hyperparamètres + métadonnées

    Et met à jour runs_history.json pour le Hall of Fame.
    """
    _ensure_dirs()
    rdir = _run_dir(run_id)
    os.makedirs(rdir, exist_ok=True)

    # --- 1) Modèle (sans pickler l'objet Network) ---
    model_payload = {
        "sizes": net.sizes,
        "weights": net.weights,
        "biases": net.biases,
    }
    with open(os.path.join(rdir, "model.pkl"), "wb") as f:
        pickle.dump(model_payload, f)

    # --- 2) Metrics ---
    with open(os.path.join(rdir, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics_history, f)

    # --- 3) Weight history ---
    with open(os.path.join(rdir, "weights.pkl"), "wb") as f:
        pickle.dump(weight_history, f)

    # --- 4) Misclassified ---
    with open(os.path.join(rdir, "misclassified.pkl"), "wb") as f:
        pickle.dump(misclassified, f)

    # --- 5) Config + summary ---
    config_with_acc = dict(config)
    config_with_acc["final_accuracy"] = float(final_accuracy)
    with open(os.path.join(rdir, "config.pkl"), "wb") as f:
        pickle.dump(config_with_acc, f)

    # --- 6) Met à jour l’historique global (pour le Hall of Fame) ---
    history = load_runs_history()

    summary = {
        "run_id": run_id,
        "timestamp": config.get("timestamp"),
        "sizes": config.get("sizes"),
        "final_accuracy": float(final_accuracy),
        "epochs": config.get("epochs"),
        "learning_rate": config.get("learning_rate"),
        "mini_batch_size": config.get("mini_batch_size"),
        "use_validation": config.get("use_validation"),
        "limit_train": config.get("limit_train"),
    }

    # Si le run_id existe déjà dans l’historique, on le remplace
    history = [h for h in history if h.get("run_id") != run_id]
    history.append(summary)

    save_runs_history(history)


# ============================================================
# Chargement d’un run (utilisé par Hall of Fame + onglets)
# ============================================================

def load_single_run(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Charge toutes les données associées à un run.

    Retourne un dict:
      {
        "model": {...},
        "metrics": [...],
        "weight_history": {...},
        "misclassified": [...],
        "config": {...}
      }

    Si le dossier ou les fichiers manquent, retourne None.
    """
    _ensure_dirs()
    rdir = _run_dir(run_id)
    if not os.path.isdir(rdir):
        # aucun dossier pour ce run → on ne peut rien charger
        return None

    def _safe_load_pickle(path: str, default):
        if not os.path.exists(path):
            return default
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return default

    # --- Tente au cas où tu aurais un full.pkl historique ---
    full_path = os.path.join(rdir, "full.pkl")
    if os.path.exists(full_path):
        try:
            with open(full_path, "rb") as f:
                full = pickle.load(f)
            # On s'assure qu'on renvoie au moins les clés attendues
            if isinstance(full, dict):
                full.setdefault("metrics", [])
                full.setdefault("weight_history", {})
                full.setdefault("misclassified", [])
                full.setdefault("config", {})
                return full
        except Exception:
            # On tombe alors sur le mode "fichiers séparés"
            pass

    # --- Format fichiers séparés (moderne) ---
    model = _safe_load_pickle(os.path.join(rdir, "model.pkl"), None)
    if model is None:
        # sans modèle, on considère le run inutilisable
        return None

    metrics = _safe_load_pickle(os.path.join(rdir, "metrics.pkl"), [])
    weights = _safe_load_pickle(os.path.join(rdir, "weights.pkl"), {})
    miscls = _safe_load_pickle(os.path.join(rdir, "misclassified.pkl"), [])
    config = _safe_load_pickle(os.path.join(rdir, "config.pkl"), {})

    return {
        "model": model,
        "metrics": metrics,
        "weight_history": weights,
        "misclassified": miscls,
        "config": config,
    }


# ============================================================
# Chargement de tous les runs au démarrage
# ============================================================

def load_all_runs() -> Dict[str, Dict[str, Any]]:
    """
    Parcourt runs/ et charge tous les runs disponibles.

    Retourne un dict: run_id -> full_run_dict
    """
    _ensure_dirs()
    all_runs: Dict[str, Dict[str, Any]] = {}

    # On se base sur les dossiers présents
    for name in os.listdir(RUNS_DIR):
        rdir = os.path.join(RUNS_DIR, name)
        if not os.path.isdir(rdir):
            continue

        full = load_single_run(name)
        if full:
            all_runs[name] = full

    return all_runs
