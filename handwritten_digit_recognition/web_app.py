# ===============================================================
#  web_app.py — Bloc 1 / 6
#  Imports, configuration, chargement des données, persistance
# ===============================================================

import os
import pickle
from datetime import datetime

import numpy as np
import streamlit as st
import cv2
from streamlit_drawable_canvas import st_canvas

# --- Modules internes ---
from persistence import (
    load_all_runs,
    load_runs_history,
    save_runs_history,
    save_full_run,
    get_hof_top3,
)
from training_lock import (
    try_acquire_lock,
    release_lock,
    is_training_in_progress,
)

import mnist_loader
from neural_network import Network


# ===============================================================
#    CSS — Ajuste la hauteur du canvas
# ===============================================================

CANVAS_HEIGHT = 400

st.markdown(
    f"""
    <style>
    iframe[title="streamlit_drawable_canvas.st_canvas"] {{
        height: {CANVAS_HEIGHT}px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ===============================================================
#    CONFIG STREAMLIT
# ===============================================================

st.set_page_config(
    page_title="MNIST Lab – DL Playground",
    layout="wide"
)

st.title("🧠 MNIST Deep Learning Lab")
st.caption("Petit labo interactif pour explorer ton réseau neuronal MNIST.")


# ===============================================================
#    CHARGEMENT DES DONNÉES MNIST
# ===============================================================

@st.cache_resource(show_spinner=True)
def load_data():
    train, val, test = mnist_loader.load_data_wrapper()
    return list(train), list(val), list(test)

training_data, validation_data, test_data = load_data()


# ===============================================================
#    CHARGEMENT DE TOUS LES RUNS (persistence)
# ===============================================================

if "all_runs" not in st.session_state:
    st.session_state.all_runs = load_all_runs()

if "runs_history" not in st.session_state:
    st.session_state.runs_history = load_runs_history()


# ===============================================================
#    ÉTAT GLOBAL STREAMLIT
# ===============================================================

if "log_text" not in st.session_state:
    st.session_state.log_text = ""

if "current_run" not in st.session_state:
    st.session_state.current_run = None

if "metrics_history" not in st.session_state:
    st.session_state.metrics_history = []

if "weight_history" not in st.session_state:
    st.session_state.weight_history = {}

if "misclassified_cache" not in st.session_state:
    st.session_state.misclassified_cache = {}

if "log_placeholder" not in st.session_state:
    st.session_state.log_placeholder = None


# ===============================================================
#    UTILITAIRES
# ===============================================================

def append_log(msg: str):
    st.session_state.log_text += msg + "\n"

    placeholder = st.session_state.get("log_placeholder", None)
    if placeholder is not None:
        placeholder.text_area("Output", st.session_state.log_text, height=400)


def clear_log():
    st.session_state.log_text = ""


def softmax(a):
    a = a - np.max(a)
    exp = np.exp(a)
    return exp / np.sum(exp)


def forward_with_activations(net: Network, x):
    activations = [x]
    zs = []
    a = x
    for b, w in zip(net.biases, net.weights):
        z = np.dot(w, a) + b
        zs.append(z)
        a = net.sigmoid(z)
        activations.append(a)
    return zs, activations


def get_misclassified(net: Network, test_data, max_samples=32):
    samples = []
    for x, y in test_data:
        a = net.feedForward(x)
        y_pred = int(np.argmax(a))
        y_true = int(y)
        if y_pred != y_true:
            probs = softmax(a)
            samples.append((x, y_true, y_pred, probs))
            if len(samples) >= max_samples:
                break
    return samples


def compute_weight_stats(net: Network):
    weights = np.concatenate([w.ravel() for w in net.weights])
    return {
        "mean": float(np.mean(weights)),
        "std": float(np.std(weights)),
        "min": float(np.min(weights)),
        "max": float(np.max(weights)),
    }

# ===============================================================
#   web_app.py — Bloc 2 / 6
#   Sidebar, onglets, Hall of Fame, chargement modèle actif
# ===============================================================


# ===============================================================
#   AFFICHAGE DU MODÈLE ACTIF
# ===============================================================

if st.session_state.current_run is None:
    st.info("🎯 Aucun modèle actif. Entraîne un modèle ou sélectionne-en un dans le Hall of Fame.")
else:
    run = st.session_state.current_run
    st.success(
        f"**Modèle actif :** `{run['run_id']}` — "
        f"({run.get('source', 'training')}) — "
        f"Acc: {run.get('final_accuracy', 0)*100:.2f}%"
    )


# ===============================================================
#   SIDEBAR – CONTRÔLES UTILISATEUR
# ===============================================================

st.sidebar.header("🎛️ Hyperparamètres")

epochs = st.sidebar.slider("Epochs", 1, 50, 10)
learning_rate = st.sidebar.slider("Learning rate (η)", 0.01, 5.0, 3.0, step=0.01)
mini_batch_size = st.sidebar.slider("Mini-batch size", 1, 100, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("🧱 Architecture")

hidden_size = st.sidebar.slider("Taille couche cachée", 10, 300, 100)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Options d'entraînement")

use_validation = st.sidebar.checkbox("Utiliser validation comme test", value=False)
limit_train = st.sidebar.number_input(
    "Limiter le nb d'exemples d'entraînement (0 = tout)",
    min_value=0,
    max_value=len(training_data),
    value=0,
    step=1000,
)


# ===============================================================
#   STRUCTURE DES ONGLET
# ===============================================================

(
    tab_readme,
    tab_train,
    tab_draw,
    tab_activations,
    tab_weights,
    tab_metrics,
    tab_errors
) = st.tabs([
    "📖 Readme",
    "🚀 Entraînement",
    "🖊️ Dessiner & Tester",
    "✨ Activations",
    "🧮 Poids",
    "📈 Métriques",
    "🕵️ Erreurs"
])


# ===============================================================
#   HALL OF FAME (TOP 3)
# ===============================================================

def draw_hall_of_fame():
    st.markdown("## 🏆 Hall of Fame (Top 3)")
    history = st.session_state.runs_history

    if not history:
        st.info("Aucun modèle enregistré pour l’instant.")
        return

    top3 = get_hof_top3(history)

    cols = st.columns(len(top3))

    for i, entry in enumerate(top3):
        with cols[i]:
            st.markdown(
                f"""
                ### 🥇 #{i+1}
                **Run ID :** `{entry['run_id']}`  
                **Accuracy :** `{entry['final_accuracy']*100:.2f}%`  
                **Hidden size :** `{entry['sizes'][1]}`  
                **Date :** `{entry['timestamp']}`
                """
            )

            if st.button(f"➡️ Utiliser {entry['run_id']}", key=f"hof_load_{i}"):
                # Charger un modèle depuis persistence
                full = st.session_state.all_runs.get(entry["run_id"])
                if full:
                    model_dict = full["model"]
                    net = Network(model_dict["sizes"])
                    net.weights = model_dict["weights"]
                    net.biases = model_dict["biases"]

                    # Mise à jour du modèle actif
                    st.session_state.current_run = {
                        "run_id": entry["run_id"],
                        "sizes": entry["sizes"],
                        "timestamp": entry["timestamp"],
                        "final_accuracy": entry["final_accuracy"],
                        "model_path": entry["model_path"],
                        "source": "hall_of_fame",
                    }

                    # On recharge les caches associées
                    st.session_state.metrics_history = full["metrics"]
                    st.session_state.weight_history = {
                        entry["run_id"]: full["weight_history"]
                    }
                    st.session_state.misclassified_cache = {
                        entry["run_id"]: full["misclassified"]
                    }

                    st.success(f"Modèle {entry['run_id']} chargé depuis Hall of Fame !")
                    st.rerun()


# ===============================================================
#   web_app.py — Bloc 3 / 6
#   Entraînement, callbacks, verrou global, sauvegarde persistante
# ===============================================================


# ===============================================================
#   CALLBACK D’ENTRAÎNEMENT (Epoch → Metrics + Weight History)
# ===============================================================

def make_epoch_callback(run_id):
    """
    Fonction appelée à chaque epoch : enregistre les métriques et les poids.
    """
    def epoch_callback(epoch, metrics, network: Network):

        # --- Sauvegarde des métriques dans state ---
        entry = {"run_id": run_id, "epoch": epoch}
        entry.update(metrics)
        st.session_state.metrics_history.append(entry)

        # --- Sauvegarde des poids (première couche) ---
        if run_id not in st.session_state.weight_history:
            st.session_state.weight_history[run_id] = {
                "epochs": [],
                "w0_list": [],
            }

        hist = st.session_state.weight_history[run_id]
        hist["epochs"].append(epoch)

        # copie pour ne pas être écrasé
        w0 = network.weights[0].copy()
        hist["w0_list"].append(w0)

    return epoch_callback


# ===============================================================
#   FONCTION PRINCIPALE DE TRAINING
# ===============================================================

def run_single_training(eta, batch, cfg_name="manual"):
    """
    Démarre un entraînement complet :
    - Acquisition du verrou global
    - Création du run_id
    - Exécute le SGD avec callbacks
    - Sauvegarde modèle + metrics + erreurs + poids
    - Libère le verrou
    """

    # -----------------------------------------------------------
    # 🔒 VERROU GLOBAL : on vérifie que personne ne s’entraîne
    # -----------------------------------------------------------
    if not try_acquire_lock():
        st.error("🚫 Un entraînement est déjà en cours dans une autre session.")
        return None, None

    try:
        clear_log()

        # Choix du dataset d'entraînement (limite optionnelle)
        if limit_train and limit_train > 0:
            train_sub = training_data[:limit_train]
        else:
            train_sub = training_data

        test_set = validation_data if use_validation else test_data

        # --- Créer réseau ---
        sizes = [784, hidden_size, 10]
        net = Network(sizes)

        # Avant entraînement on ne connaît pas l’acc → on met juste model number + date
        model_number = len(st.session_state.runs_history) + 1  # auto-incrément
        run_id = f"M{model_number}_{datetime.now().strftime('%y%m%d')}"

        # --- Metadonnées en mémoire ---
        st.session_state.current_run = {
            "run_id": run_id,
            "sizes": sizes,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config_name": cfg_name,
            "source": "training",
        }

        append_log(f"=== NEW RUN {run_id} ===")
        append_log(f"Architecture: {sizes}")
        append_log(f"Epochs={epochs}, eta={eta}, mini_batch={batch}")
        append_log(f"Train samples={len(train_sub)}, Test samples={len(test_set)}")

        # --- Entraînement SGD ---
        net.SGD(
            training_data=train_sub,
            epochs=epochs,
            mini_batch_size=batch,
            eta=eta,
            test_data=test_set,
            log_fn=append_log,
            epoch_callback=make_epoch_callback(run_id),
        )

        # --- Évaluer modèle final ---
        correct = net.evaluate(test_set)
        final_acc = correct / len(test_set)
        st.session_state.current_run["final_accuracy"] = final_acc

        append_log(f"Final accuracy: {final_acc:.4f}")

        # --- Erreurs (misclassified) ---
        miscls = get_misclassified(net, test_set)
        st.session_state.misclassified_cache[run_id] = miscls

        # --- Sauvegarde persistante complète ---
        full_config = {
            "timestamp": st.session_state.current_run["timestamp"],
            "sizes": sizes,
            "config_name": cfg_name,
            "epochs": epochs,
            "learning_rate": eta,
            "mini_batch_size": batch,
            "use_validation": use_validation,
            "limit_train": limit_train,
        }

        save_full_run(
            run_id=run_id,
            net=net,
            final_accuracy=final_acc,
            metrics_history=[
                m for m in st.session_state.metrics_history if m["run_id"] == run_id
            ],
            weight_history=st.session_state.weight_history.get(run_id, {}),
            misclassified=miscls,
            config=full_config,
        )

        # Rajouter au cache global des runs
        st.session_state.all_runs[run_id] = {
            "model": {
                "sizes": sizes,
                "weights": net.weights,
                "biases": net.biases
            },
            "metrics": [
                m for m in st.session_state.metrics_history if m["run_id"] == run_id
            ],
            "weight_history": st.session_state.weight_history.get(run_id, {}),
            "misclassified": miscls,
            "config": full_config,
        }

        # Rajouter dans runs_history display
        st.session_state.runs_history = load_runs_history()

        append_log("✔️ Entraînement terminé et sauvegardé.")

        return run_id, net

    finally:
        # 🔓 Libération du verrou global
        release_lock()
        append_log("Lock released.")

# ===============================================================
#   web_app.py — Bloc 4 / 6
#   Interface de l’onglet Training : terminal, bouton start,
#   Hall of Fame, affichage du dernier run
# ===============================================================

# ==============================================================
#   ONGLET README
# ==============================================================

with tab_readme:
    st.subheader("Bienvenue dans le MNIST Deep Learning Lab 👋")

    st.markdown("""
## 📚 Qu’est-ce que MNIST ?

MNIST, c’est un petit classique du machine learning.  
Il s’agit d’un jeu de données contenant **70 000 images de chiffres manuscrits** (de 0 à 9), chacune en **28×28 pixels**.  
Les images proviennent de milliers de personnes différentes, ce qui en fait un terrain parfait pour apprendre comment un modèle reconnaît des motifs visuels.

En bref :  
> MNIST, c’est le *“Hello World”* du Deep Learning — simple, propre, et idéal pour comprendre les bases.

---

## 🎯 1. À quoi sert cette application ?

Ce site te permet de **configurer**, **entraîner** et **tester** ton propre réseau de neurones, le tout sans écrire une seule ligne de code.

Tu peux :

### 🔧 Configurer ton réseau
- Choisir la taille de la couche cachée  
- Ajuster les hyperparamètres (epochs, learning rate, mini-batch…)  
- Activer certaines options d’entraînement  

### 🚀 Lancer l’entraînement
- Suivre la progression dans un terminal en direct  
- Visualiser l’évolution de l’accuracy  
- Voir les erreurs, les activations internes et même les poids appris par le modèle  

### ✏️ Tester le modèle
- Sur des images MNIST réelles  
- Ou en dessinant toi-même un chiffre dans un canvas interactif

L’objectif est pédagogique : comprendre *comment* un réseau apprend, et *pourquoi* il se trompe parfois.

---

## 🧩 2. Comment fonctionne un réseau de neurones ? (Version simple)

Un réseau de neurones, c’est un ensemble de “couches” qui transforment progressivement une entrée (ici, une image 28×28) pour prédire un chiffre.

### Structure typique :
- **Input** : 784 pixels (28×28)
- **Hidden layer** : une couche de neurones intermédiaires
- **Output** : 10 neurones (un par chiffre 0–9)

À chaque étape :

1. Les neurones reçoivent des nombres (les intensités des pixels)
2. Ils les multiplient par des **poids**
3. Ils appliquent une fonction (sigmoïde)
4. Ils transmettent le résultat à la couche suivante

Pendant l’entraînement, le modèle :

- fait une prédiction  
- mesure l’erreur  
- ajuste ses poids pour faire mieux au prochain passage  

En répétant ça des milliers de fois → il apprend.

---

## ⚙️ 3. Les hyperparamètres : ce qu’ils font, et comment les régler

Les hyperparamètres sont les réglages qui influencent *comment* le modèle apprend.

### 🔸 **Epochs**
Le nombre de fois où le modèle passe sur **tout** le dataset.

- Peu : modèle pas assez entraîné  
- Trop : risque de mémoriser inutilement  

💡 Pour MNIST : **10 à 30 epochs suffisent largement**

---

### 🔸 **Learning rate (η)**
La “vitesse d’apprentissage”.

- Trop faible → apprentissage lent  
- Trop fort → instable, le modèle oscille ou diverge  

💡 Pour ce réseau : **entre 0.5 et 3.0 fonctionne très bien**

---

### 🔸 **Mini-batch size**
Nombre d’exemples utilisés avant chaque mise à jour des poids.

- Petit batch → apprentissage plus “vivant”, mais plus bruité  
- Gros batch → plus stable, mais peut donner des résultats moins bons  

💡 Valeurs conseillées : **10 à 50**

---

## 🧱 4. La couche cachée (Hidden Layer)

La couche cachée est le cœur du modèle : c’est là qu’il apprend les **motifs** caractéristiques des chiffres :

- courbes  
- angles  
- tiges verticales  
- coins  
- boucles  
- etc.

Plus la hidden layer est grande :

- plus le modèle peut apprendre de choses  
- mais plus il devient lent, et plus il risque de surapprendre

💡 Pour MNIST : **entre 50 et 150 neurones**, c’est un bon compromis.

---

## 🧪 5. Les options d’entraînement

### 🔸 Utiliser la validation comme test
Permet d’évaluer le modèle *sans toucher au vrai jeu de test*.  
C’est pratique pour ajuster les hyperparamètres sans “tricher” sur les performances réelles.

### 🔸 Limiter le nombre d’exemples d’entraînement
Tu peux choisir de n’entraîner le modèle que sur une partie du dataset.

Utile pour :
- des tests rapides  
- économiser les ressources  
- observer comment la quantité de données influence l’apprentissage  

💡 0 = utiliser tout MNIST (valeur normale)

---

## 🚀 6. Lancer l’entraînement

1. Choisis :
   - la taille de la hidden layer  
   - les hyperparamètres  
   - les options d’entraînement  

2. Clique sur **Start training**

3. Observe :
   - le terminal qui se met à jour  
   - les courbes d’évolution  
   - les erreurs et activations internes  
   - les poids du réseau  

À la fin, un modèle est automatiquement sauvegardé.

---

## ✏️ 7. Tester ton modèle (canvas de dessin)

Dans l’onglet **🖊️ Dessiner & Tester** :

- sélectionne un modèle sauvegardé  
- dessine un chiffre à la souris  

Le dessin est automatiquement :

- converti en image 28×28  
- normalisé  
- passé au modèle

Le réseau te renvoie :
- sa prédiction  
- les probabilités associées (softmax)

---

N’hésite pas à explorer, tester plusieurs hyperparamètres et comparer les résultats.  
Amuse-toi bien avec le Deep Learning 🙂
""")


# ===============================================================
#   ONGLET TRAINING
# ===============================================================

with tab_train:

    # -----------------------------------------------------------
    #   COLONNE GAUCHE : Démarrer un entraînement
    # -----------------------------------------------------------
    col_left, col_right = st.columns([2, 1])

    with col_left:

        st.subheader("🚀 Lancer un nouvel entraînement")

        # Bouton Start Training
        start_training = st.button("🔥 Start Training", disabled=is_training_in_progress())

        if is_training_in_progress():
            st.warning("⏳ Un entraînement est déjà en cours. Veuillez patienter...")

        st.markdown("### 📡 Terminal")

        # Création du placeholder si inexistant
        if st.session_state.log_placeholder is None:
            st.session_state.log_placeholder = st.empty()

        # Affichage du terminal
        st.session_state.log_placeholder.text_area(
            "Output",
            value=st.session_state.log_text,
            height=400,
        )


    # -----------------------------------------------------------
    #   COLONNE DROITE : Dernier run
    # -----------------------------------------------------------
    with col_right:
        st.subheader("📝 Dernier run")

        run = st.session_state.current_run

        if run is None:
            st.info("Aucun run pour le moment.")
        else:
            st.write(f"**Run ID :** `{run['run_id']}`")
            st.write(f"**Date :** {run['timestamp']}`")
            st.write(f"**Architecture :** `{run['sizes']}`")

            if "final_accuracy" in run:
                st.metric("Accuracy test", f"{run['final_accuracy']*100:.2f} %")
            else:
                st.info("Aucune accuracy disponible (run en cours ?).")

        st.markdown("---")

        # Hall of Fame
        draw_hall_of_fame()


    # -----------------------------------------------------------
    #   LANCEMENT D’UN RUN
    # -----------------------------------------------------------
    if start_training and not is_training_in_progress():
        # On réinitialise les structures par run
        st.session_state.metrics_history = [
            m for m in st.session_state.metrics_history
            if m["run_id"].startswith("dummy_never")
        ]
        st.session_state.weight_history = {}
        st.session_state.misclassified_cache = {}

        # Lancer l'entraînement
        run_single_training(
            eta=learning_rate,
            batch=mini_batch_size,
            cfg_name="manual"
        )

        # Forcer rafraîchissement Streamlit pour afficher les résultats
        st.rerun()

# ===============================================================
#   web_app.py — Bloc 5 / 6
#   Métriques (graphs), Poids, Activations, Erreurs,
#   TSNE / PCA, Matrice de confusion
# ===============================================================


# ===============================================================
#     ONGLET MÉTRIQUES
# ===============================================================

with tab_metrics:
    st.subheader("📈 Métriques d'entraînement et performance")

    if not st.session_state.metrics_history:
        st.info("Aucune métrrique disponible. Entraîne un modèle.")
    else:
        import pandas as pd

        df = pd.DataFrame(st.session_state.metrics_history)
        run_ids = df["run_id"].unique().tolist()

        selected_run = st.selectbox("Sélectionne un run", run_ids)
        df_run = df[df["run_id"] == selected_run].sort_values("epoch")

        # === LIGNE DE 2 COLONNES ===
        col1, col2 = st.columns(2)

        # === COURBE D'ACCURACY ===
        with col1:
            st.markdown("### Accuracy par epoch")
            if "test_accuracy" in df_run:
                st.line_chart(
                    df_run.set_index("epoch")["test_accuracy"],
                    height=300
                )

        # === COURBE DE NB DE PRÉDICTIONS CORRECTES ===
        with col2:
            st.markdown("### Prédictions correctes par epoch")
            if "test_correct" in df_run:
                st.bar_chart(
                    df_run.set_index("epoch")["test_correct"],
                    height=300
                )


    st.markdown("---")

    with st.expander("🧠 Loss par epoch"):

        st.markdown(
            """
        La **loss** mesure à quel point le modèle se trompe en moyenne.

        - Après chaque epoch, on calcule une valeur de loss sur le jeu d'entraînement.
        - Normalement, la loss doit **descendre** progressivement si le modèle apprend correctement.
        """
        )

        if not st.session_state.metrics_history:
            st.info("Aucune métrique disponible.")
        else:
            df = pd.DataFrame(st.session_state.metrics_history)
            run_ids = df["run_id"].unique().tolist()

            selected_run_loss = st.selectbox(
                "Sélectionner un run",
                run_ids,
                key="loss_selector"
            )

            df_run_loss = df[df["run_id"] == selected_run_loss].sort_values("epoch")

            if "train_loss" in df_run_loss:
                st.line_chart(
                    df_run_loss.set_index("epoch")["train_loss"],
                    height=300
                )

                st.metric(
                    "Dernière loss enregistrée",
                    f"{df_run_loss['train_loss'].iloc[-1]:.4f}"
                )


    # ============================================================
    #   🧩 TSNE / PCA des activations internes
    # ============================================================

    with st.expander("🧩 Visualisation TSNE / PCA de la hidden layer"):
        st.markdown(
            """
            On projette ici les activations de la **couche cachée** dans un plan 2D
            pour voir comment le réseau sépare les chiffres dans son espace interne.

            Chaque point = une image MNIST, colorée selon son vrai chiffre.
            """
        )

        if st.session_state.current_run is None:
            st.info("Aucun modèle actif.")
        else:
            run_id = st.session_state.current_run["run_id"]
            full = st.session_state.all_runs.get(run_id)

            if not full:
                st.error("Impossible de charger les données du modèle.")
            else:
                model_dict = full["model"]
                net = Network(model_dict["sizes"])
                net.weights = model_dict["weights"]
                net.biases = model_dict["biases"]

                import pandas as pd
                import altair as alt
                from sklearn.decomposition import PCA
                from sklearn.manifold import TSNE

                max_samples = len(test_data)

                n_samples = st.slider(
                    "Nombre d'images",
                    100, min(2000, max_samples),
                    500
                )
                method = st.radio(
                    "Méthode",
                    ["PCA", "t-SNE"]
                )

                if st.button("Calculer projection 2D"):
                    X = []
                    y_labels = []

                    for i, (x, y_true) in enumerate(test_data[:n_samples]):
                        _, acts = forward_with_activations(net, x)
                        hidden = acts[1]
                        X.append(hidden.ravel())
                        y_labels.append(int(y_true))

                    X = np.array(X)

                    if method == "PCA":
                        reducer = PCA(n_components=2)
                    else:
                        reducer = TSNE(
                            n_components=2,
                            learning_rate="auto",
                            init="random",
                            perplexity=min(30, n_samples - 1),
                        )

                    with st.spinner("Calcul..."):
                        emb = reducer.fit_transform(X)

                    df_emb = pd.DataFrame({
                        "x": emb[:, 0],
                        "y": emb[:, 1],
                        "label": y_labels,
                    })

                    chart = alt.Chart(df_emb).mark_circle(size=50, opacity=0.8).encode(
                        x="x",
                        y="y",
                        color="label:N",
                        tooltip=["label"]
                    )

                    st.altair_chart(chart, use_container_width=True)

    # ============================================================
    #   🎞️ Animation de l'évolution des poids d'un neurone
    # ============================================================

    with st.expander("🎞️ Animation des poids d'un neurone (hidden layer)"):

        st.markdown(
            """
            Chaque neurone de la couche cachée possède **784 poids** (un par pixel).  
            Si on reshape ce vecteur en 28×28, on obtient une image qui représente **le motif auquel ce neurone est sensible**.

            Idée de visualisation :
            - pour un neurone donné,
            - on enregistre ses poids à différents epochs,
            - puis on affiche une **série d’images** (ou un slider temporel) qui montre comment ce motif évolue.

            Ce que l’on voit :
            - au début, les poids ressemblent à du bruit ;
            - progressivement, des formes apparaissent (traits verticaux, courbes, zones sombres/claires) ;
            - le neurone se “spécialise” dans un type de motif.

            > C’est une excellente manière d’illustrer qu’un réseau n’est pas une boîte noire magique, mais qu’il apprend effectivement des patrons visuels.
            """
        )

        if not st.session_state.weight_history:
            st.info("Aucun historique de poids disponible.")
        else:
            run_ids_w = list(st.session_state.weight_history.keys())
            sel_run = st.selectbox("Sélectionner un run", run_ids_w)

            hist = st.session_state.weight_history[sel_run]
            epochs_hist = hist["epochs"]
            w0_list = hist["w0_list"]

            hidden_size_hist = w0_list[0].shape[0]

            neuron_idx = st.slider(
                "Neurone",
                0, hidden_size_hist - 1,
                0
            )
            epoch_pos = st.slider(
                "Position epoch",
                0, len(epochs_hist) - 1,
                len(epochs_hist) - 1
            )

            w_vec = w0_list[epoch_pos][neuron_idx, :]
            img = w_vec.reshape(28, 28)

            mn, mx = img.min(), img.max()
            if mx > mn:
                img_norm = (img - mn) / (mx - mn)
            else:
                img_norm = np.zeros_like(img)

            st.image(
                img_norm,
                width=160,
                caption=f"Run {sel_run} — neurone {neuron_idx}, epoch {epochs_hist[epoch_pos]}"
            )


    # ============================================================
    #   🧮 Matrice de confusion
    # ============================================================

    with st.expander("🧮 Matrice de confusion"):

        st.markdown(
            """
        La **matrice de confusion** résume comment le modèle se trompe entre les classes.

        - en lignes : la *vraie* classe (0, 1, 2, …, 9)
        - en colonnes : la classe *prédite* par le modèle
        - chaque case contient le nombre d’exemples correspondant

        On s’attend à ce que la **diagonale** soit dominante (bonnes prédictions).
        """
        )

        if st.session_state.current_run is None:
            st.info("Aucun modèle actif.")
        else:
            run_id = st.session_state.current_run["run_id"]
            full = st.session_state.all_runs.get(run_id)

            if not full:
                st.warning("Impossible de charger le modèle.")
            else:
                model_dict = full["model"]
                net = Network(model_dict["sizes"])
                net.weights = model_dict["weights"]
                net.biases = model_dict["biases"]

                dataset_choice = st.radio(
                    "Dataset",
                    ["Test", "Validation"],
                    horizontal=True
                )
                data = test_data if dataset_choice == "Test" else validation_data

                n_samples = st.slider(
                    "Nombre d'images",
                    100,
                    len(data),
                    min(1000, len(data))
                )

                if st.button("Calculer matrice"):
                    with st.spinner("Calcul en cours..."):
                        cm = np.zeros((10, 10), dtype=int)
                        subset = data[:n_samples]

                        for x, y_true in subset:
                            a = net.feedForward(x)
                            y_pred = int(np.argmax(a))
                            y_true = int(y_true)
                            cm[y_true, y_pred] += 1

                    st.write("### Accuracy :")
                    acc = np.trace(cm) / cm.sum()
                    st.metric("Accuracy", f"{acc*100:.2f}%")

                    import pandas as pd
                    import seaborn as sns
                    import matplotlib.pyplot as plt

                    df_cm = pd.DataFrame(cm)
                    fig, ax = plt.subplots()
                    sns.heatmap(df_cm, annot=False, cmap="Blues", ax=ax)
                    st.pyplot(fig)


# ===============================================================
#     ONGLET ERREURS
# ===============================================================

with tab_errors:
    st.subheader("🕵️ Images mal classées")

    if st.session_state.current_run is None:
        st.info("Aucun modèle actif.")
    else:
        run_id = st.session_state.current_run["run_id"]
        errors = st.session_state.misclassified_cache.get(run_id)

        if not errors:
            st.info("Aucune erreur disponible.")
        else:
            cols = st.columns(8)
            for i, (x, y_true, y_pred, probs) in enumerate(errors):
                col = cols[i % len(cols)]
                img = np.reshape(x, (28, 28))
                col.image(img, width=60, caption=f"T:{y_true}/P:{y_pred}")


# ===============================================================
#     ONGLET ACTIVATIONS
# ===============================================================

with tab_activations:
    st.subheader("✨ Explorateur d'activations internes")

    st.markdown(
        """
        Cet onglet permet d’explorer ce qui se passe *à l’intérieur* du réseau de neurones
        lorsqu’il traite une image MNIST.

        À partir d’une image du jeu de test :

        - tu vois sa **distribution de probabilités** en sortie (softmax),
        - tu peux analyser les **activations de chaque couche**,
        - et visualiser la **norme des activations**, qui donne une idée de l’intensité de la réponse du réseau.

        L’objectif est de mieux comprendre comment chaque couche transforme l’information
        et comment le réseau « réagit » à une image donnée.

        En changeant l'index de l’image, tu peux comparer les activations pour différents chiffres.

        """
    )
    if st.session_state.current_run is None:
        st.info("Aucun modèle actif.")
    else:
        run_id = st.session_state.current_run["run_id"]
        full = st.session_state.all_runs.get(run_id)

        if not full:
            st.warning("Impossible de charger le modèle.")
        else:
            model_dict = full["model"]
            net = Network(model_dict["sizes"])
            net.weights = model_dict["weights"]
            net.biases = model_dict["biases"]

            idx = st.slider(
                "Index MNIST",
                0, len(test_data) - 1,
                0
            )

            x, y_true = test_data[idx]
            img = x.reshape(28, 28)

            colA, colB = st.columns([1, 2])

            with colA:
                st.image(img, width=140, caption=f"Label : {y_true}")

            with colB:
                zs, activations = forward_with_activations(net, x)
                output = activations[-1]
                probs = softmax(output)

                import pandas as pd

                df_probs = pd.DataFrame({
                    "digit": list(range(10)),
                    "proba": probs.ravel()
                })
                st.bar_chart(df_probs.set_index("digit"))

                norms = [float(np.linalg.norm(a)) for a in activations]
                df_norms = pd.DataFrame({
                    "layer": list(range(len(norms))),
                    "activation_norm": norms,
                })
                st.line_chart(df_norms.set_index("layer"))


# ===============================================================
#     ONGLET POIDS
# ===============================================================

with tab_weights:
    st.subheader("🧮 Analyse des poids")

    st.markdown(
        """
        Ici, tu visualises ce que le réseau a appris :

        - Les **poids de la couche cachée** montrent les motifs auxquels chaque neurone est sensible.
        - Les **poids de sortie** montrent comment ces motifs sont combinés pour reconnaître chaque chiffre.

        C’est une façon rapide de voir ce que le modèle “regarde” dans les images.
        """
    )

    if st.session_state.current_run is None:
        st.info("Aucun modèle actif.")
    else:
        run_id = st.session_state.current_run["run_id"]
        full = st.session_state.all_runs.get(run_id)

        if not full:
            st.warning("Impossible de charger le modèle.")
        else:
            model_dict = full["model"]
            net = Network(model_dict["sizes"])
            net.weights = model_dict["weights"]
            net.biases = model_dict["biases"]

            stats = compute_weight_stats(net)
            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Mean", f"{stats['mean']:.4e}")
            c2.metric("Std", f"{stats['std']:.4e}")
            c3.metric("Min", f"{stats['min']:.4e}")
            c4.metric("Max", f"{stats['max']:.4e}")

            st.markdown("---")
            st.markdown("### Poids input → hidden")

            w0 = net.weights[0]
            cols = st.columns(10)
            for i in range(min(10, w0.shape[0])):
                col = cols[i % len(cols)]
                img = w0[i, :].reshape(28, 28)

                mn, mx = img.min(), img.max()
                if mx > mn:
                    img_norm = (img - mn) / (mx - mn)
                else:
                    img_norm = np.zeros_like(img)

                col.image(img_norm, width=60, caption=f"N {i}")

            st.markdown("---")
            st.markdown("### Poids hidden → output")

            w_out = net.weights[-1]
            w_hidden = net.weights[0]

            cols = st.columns(10)
            for digit in range(10):
                col = cols[digit % len(cols)]

                combined = np.dot(w_out[digit], w_hidden)
                img = combined.reshape(28, 28)

                mn, mx = img.min(), img.max()
                if mx > mn:
                    img_norm = (img - mn) / (mx - mn)
                else:
                    img_norm = np.zeros_like(img)

                col.image(img_norm, width=60, caption=f"Classe {digit}")

# ===============================================================
#   web_app.py — Bloc 6 / 6
#   Onglet : 🖊️ Dessiner & Tester
# ===============================================================

with tab_draw:
    st.subheader("🖊️ Dessine un chiffre (0–9) et teste le modèle")

    st.markdown("""
    Dessine un chiffre dans la zone ci-dessous.  
    Il sera automatiquement converti en **image MNIST 28×28**, normalisé,
    puis envoyé au **modèle actif**.
    """)

    # -----------------------------------------------------------
    #  Choix du modèle actif (runs persistés)
    # -----------------------------------------------------------

    all_models = list(st.session_state.all_runs.keys())

    if not all_models:
        st.warning("Aucun modèle sauvegardé. Entraîne un modèle pour commencer.")
    else:
        # Préselection : le modèle actif actuel
        default_index = (
            all_models.index(st.session_state.current_run["run_id"])
            if st.session_state.current_run and st.session_state.current_run["run_id"] in all_models
            else 0
        )

        selected_run_id = st.selectbox(
            "Sélectionne un modèle à utiliser",
            all_models,
            index=default_index
        )

        # Charger ce modèle
        model_data = st.session_state.all_runs[selected_run_id]["model"]
        net = Network(model_data["sizes"])
        net.weights = model_data["weights"]
        net.biases = model_data["biases"]

        # -----------------------------------------------------------
        #  Canvas
        # -----------------------------------------------------------

        st.markdown("### 📝 Zone de dessin")

        if "canvas_key" not in st.session_state:
            st.session_state.canvas_key = 0

        if st.button("🧽 Effacer le dessin"):
            st.session_state.canvas_key += 1

        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=400,
            width=400,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}"
        )

        # -----------------------------------------------------------
        #  Bouton de prédiction
        # -----------------------------------------------------------

        if st.button("🔍 Prédire le chiffre dessiné"):
            if canvas_result.image_data is None:
                st.error("Aucun dessin détecté.")
            else:
                img = canvas_result.image_data

                # Conversion en niveaux de gris
                img_gray = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_BGR2GRAY)

                # Redimension MNIST
                img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)

                # Normalisation (0–1)
                img_norm = img_resized / 255.0

                # Format réseau 784x1
                x = img_norm.reshape(784, 1)

                # Prédiction
                output = net.feedForward(x)
                probs = softmax(output)
                prediction = int(np.argmax(probs))

                st.markdown("## 📌 Résultat")

                colA, colB = st.columns([1, 2])

                with colA:
                    st.image(
                        img_resized,
                        width=150,
                        caption="Image 28×28 envoyée au modèle"
                    )

                with colB:
                    st.success(f"**Le modèle prédit : {prediction}**")

                    import pandas as pd
                    df = pd.DataFrame({
                        "digit": list(range(10)),
                        "proba": probs.ravel(),
                    })
                    st.bar_chart(df.set_index("digit"))


