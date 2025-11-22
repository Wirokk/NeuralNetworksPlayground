import os
import json
import pickle
from datetime import datetime


# =========================
#    CONFIG PATHS
# =========================

RUNS_DIR = "runs"
RUNS_HISTORY_FILE = "runs_history.json"


# =========================
#    UTILITAIRES I/O
# =========================

def ensure_dirs():
    """Ensure required directories exist."""
    os.makedirs(RUNS_DIR, exist_ok=True)


def save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# =========================
#    RUNS HISTORY (GLOBAL)
# =========================

def load_runs_history():
    """Load runs_history.json, return [] if empty."""
    ensure_dirs()
    if not os.path.exists(RUNS_HISTORY_FILE):
        return []

    try:
        return load_json(RUNS_HISTORY_FILE)
    except Exception:
        return []


def save_runs_history(history):
    """Overwrite runs_history.json."""
    ensure_dirs()
    save_json(RUNS_HISTORY_FILE, history)


def append_run_to_history(run_meta):
    """Add a run entry to runs_history.json."""
    history = load_runs_history()
    history.append(run_meta)
    save_runs_history(history)


# =========================
#    SAVE FULL RUN
# =========================

def save_full_run(
    run_id,
    net,
    final_accuracy,
    metrics_history,
    weight_history,
    misclassified,
    config,
):
    """
    Save:
    - model.pkl
    - metrics.pkl
    - weight_history.pkl
    - misclassified.pkl
    - config.json
    - and append to runs_history.json
    """

    ensure_dirs()

    short_acc = f"{final_accuracy*100:.2f}".replace(".", "")
    run_dir = os.path.join(RUNS_DIR, f"{run_id}_{short_acc}")
    os.makedirs(run_dir, exist_ok=True)

    # ---- SAVE MODEL ----
    model_payload = {
        "sizes": net.sizes,
        "biases": net.biases,
        "weights": net.weights,
    }
    save_pickle(os.path.join(run_dir, "model.pkl"), model_payload)

    # ---- METRICS ----
    save_pickle(os.path.join(run_dir, "metrics.pkl"), metrics_history)

    # ---- WEIGHT HISTORY ----
    save_pickle(os.path.join(run_dir, "weight_history.pkl"), weight_history)

    # ---- MISCLASSIFIED ----
    save_pickle(os.path.join(run_dir, "misclassified.pkl"), misclassified)

    # ---- CONFIG ----
    config_with_acc = config.copy()
    config_with_acc["final_accuracy"] = final_accuracy
    config_with_acc["run_id"] = run_id
    save_json(os.path.join(run_dir, "config.json"), config_with_acc)

    # ---- UPDATE GLOBAL SCOREBOARD ----
    summary_entry = {
        "run_id": run_id,
        "timestamp": config["timestamp"],
        "sizes": config["sizes"],
        "final_accuracy": final_accuracy,
        "model_path": f"{RUNS_DIR}/{run_id}/model.pkl",
        "config_name": config["config_name"],
    }

    append_run_to_history(summary_entry)


# =========================
#    LOAD SINGLE RUN
# =========================

def load_single_run(run_id):
    run_dir = os.path.join(RUNS_DIR, run_id)
    if not os.path.exists(run_dir):
        return None

    out = {}

    # model
    model_path = os.path.join(run_dir, "model.pkl")
    if os.path.exists(model_path):
        out["model"] = load_pickle(model_path)

    # metrics
    metrics_path = os.path.join(run_dir, "metrics.pkl")
    if os.path.exists(metrics_path):
        out["metrics"] = load_pickle(metrics_path)
    else:
        out["metrics"] = []

    # weight history
    weight_path = os.path.join(run_dir, "weight_history.pkl")
    if os.path.exists(weight_path):
        out["weight_history"] = load_pickle(weight_path)
    else:
        out["weight_history"] = {}

    # misclassified
    miscls_path = os.path.join(run_dir, "misclassified.pkl")
    if os.path.exists(miscls_path):
        out["misclassified"] = load_pickle(miscls_path)
    else:
        out["misclassified"] = []

    # config
    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        out["config"] = load_json(config_path)
    else:
        out["config"] = {}

    return out


# =========================
#    LOAD ALL RUNS
# =========================

def load_all_runs():
    """
    Returns:
    {
        run_id: {
            "model": ...,
            "metrics": [...],
            "weight_history": ...,
            "misclassified": [...],
            "config": {...}
        },
        ...
    }
    """

    ensure_dirs()

    runs = {}
    for run_id in os.listdir(RUNS_DIR):
        full_dir = os.path.join(RUNS_DIR, run_id)
        if not os.path.isdir(full_dir):
            continue

        data = load_single_run(run_id)
        if data:
            runs[run_id] = data

    return runs


# =========================
#    HELPER : HALL OF FAME
# =========================

def get_hof_top3(runs_history):
    """
    Return top 3 runs based on final_accuracy.
    """
    sorted_runs = sorted(
        runs_history,
        key=lambda r: r.get("final_accuracy", 0.0),
        reverse=True
    )
    return sorted_runs[:3]
