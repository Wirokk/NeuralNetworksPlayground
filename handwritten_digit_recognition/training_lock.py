import streamlit as st
from datetime import datetime


# =============================
#     VERROU GLOBAL STREAMLIT
# =============================
# Un seul entraînement en même temps,
# toutes sessions confondues.


@st.cache_resource
def get_training_lock():
    """
    Retourne un objet mutable partagé entre TOUTES les sessions Streamlit.
    Le lock est un simple dict :
    {
        "busy": bool,
        "owner": session_id ou datetime,
    }
    """
    return {"busy": False, "owner": None}


def try_acquire_lock():
    """
    Tentative de prise du lock global.
    Si un entraînement est déjà en cours → return False.
    Sinon, prend le lock et renvoie True.
    """
    lock = get_training_lock()

    if lock["busy"]:
        return False

    lock["busy"] = True
    lock["owner"] = datetime.now().isoformat()
    return True


def release_lock():
    """
    Libère le lock global.
    """
    lock = get_training_lock()
    lock["busy"] = False
    lock["owner"] = None


def is_training_in_progress():
    """
    Retourne True si un run est actuellement en cours.
    """
    lock = get_training_lock()
    return lock["busy"]
