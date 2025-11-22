import os
import pickle
from datetime import datetime

import numpy as np
import streamlit as st

import mnist_loader
from neural_network import Network

import cv2
from streamlit_drawable_canvas import st_canvas

# =========================
#   CSS POUR FORCER LA TAILLE DU CANVAS
# =========================

CANVAS_HEIGHT = 400  # même valeur que dans st_canvas

st.markdown(
    f"""
    <style>
    /* Force la hauteur de l'iframe du composant drawable canvas */
    iframe[title="streamlit_drawable_canvas.st_canvas"] {{
        height: {CANVAS_HEIGHT}px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
#   BOOTSTRAP MODELS SAUVÉS
# =========================

def bootstrap_saved_models():
    if "saved_models" not in st.session_state:
        st.session_state.saved_models = {}

    os.makedirs("saved_models", exist_ok=True)

    for fname in os.listdir("saved_models"):
        if fname.endswith(".pkl"):
            path = os.path.join("saved_models", fname)
            run_id = fname[:-4]  # tu peux parser mieux si tu veux extraire l'accuracy
            if run_id not in st.session_state.saved_models:
                st.session_state.saved_models[run_id] = path

bootstrap_saved_models()




# ==========================
#   CONFIG STREAMLIT
# ==========================

st.set_page_config(
    page_title="MNIST Lab – DL Playground",
    layout="wide"
)

st.title("🧠 MNIST Deep Learning Lab")
st.caption("Petit labo interactif pour explorer ton réseau neuronal MNIST.")

# ==========================
#   CHARGEMENT DES DONNÉES
# ==========================

@st.cache_resource(show_spinner=True)
def load_data():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    return list(training_data), list(validation_data), list(test_data)

training_data, validation_data, test_data = load_data()

# ==========================
#   STATE GLOBAL
# ==========================

if "log_text" not in st.session_state:
    st.session_state.log_text = ""

if "metrics_history" not in st.session_state:
    # liste de dicts {run_id, epoch, test_accuracy, ...}
    st.session_state.metrics_history = []

if "current_run" not in st.session_state:
    st.session_state.current_run = None

if "misclassified_cache" not in st.session_state:
    st.session_state.misclassified_cache = {}

if "saved_models" not in st.session_state:
    # run_id -> path modèle
    st.session_state.saved_models = {}

if "weight_history" not in st.session_state:
    # run_id -> {
    #   "epochs": [1, 2, 3, ...],
    #   "w0_list": [W0_epoch1, W0_epoch2, ...]  (poids de la première couche)
    # }
    st.session_state.weight_history = {}

# ==========================
#   FONCTIONS UTILITAIRES
# ==========================

def append_log(msg: str):
    # On met à jour le buffer texte
    st.session_state.log_text += msg + "\n"

    # Si on a un placeholder pour le terminal, on le met à jour en live
    placeholder = st.session_state.get("log_placeholder", None)
    if placeholder is not None:
        placeholder.text_area(
            "Output",
            value=st.session_state.log_text,
            height=400,
        )

def clear_log():
    st.session_state.log_text = ""

def softmax(a):
    a = a - np.max(a)
    exp = np.exp(a)
    return exp / np.sum(exp)

def forward_with_activations(net: Network, x):
    """
    Retourne (zs, activations) pour toutes les couches.
    """
    activation = x
    activations = [x]
    zs = []
    for b, w in zip(net.biases, net.weights):
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = net.sigmoid(z)
        activations.append(activation)
    return zs, activations

def get_misclassified(net: Network, test_data, max_samples=32):
    """
    Retourne une liste de tuples (x, y_true, y_pred, probs)
    """
    samples = []
    for x, y in test_data:
        a = net.feedForward(x)
        probs = softmax(a)
        y_pred = int(np.argmax(probs))
        y_true = int(y)
        if y_pred != y_true:
            samples.append((x, y_true, y_pred, probs))
            if len(samples) >= max_samples:
                break
    return samples

def compute_weight_stats(net: Network):
    """
    Retourne des stats simples sur les poids.
    """
    all_weights = np.concatenate([w.ravel() for w in net.weights])
    return {
        "mean": float(np.mean(all_weights)),
        "std": float(np.std(all_weights)),
        "min": float(np.min(all_weights)),
        "max": float(np.max(all_weights)),
    }

def save_model(net: Network, run_id: str, accuracy: float = None) -> str:
    """
    Sauvegarde seulement les paramètres du réseau (sizes, biais, poids),
    et pas l'objet Network complet, pour éviter les problèmes de pickle
    avec les reruns Streamlit.
    """
    os.makedirs("saved_models", exist_ok=True)

    filename = (
        f"{run_id}_{accuracy:.4f}.pkl" if accuracy is not None else f"{run_id}.pkl"
    )
    path = os.path.join("saved_models", filename)

    payload = {
        "sizes": net.sizes,
        "biases": net.biases,
        "weights": net.weights,
    }

    with open(path, "wb") as f:
        pickle.dump(payload, f)

    st.session_state.saved_models[run_id] = path
    return path


def load_model(path: str) -> Network:
    """
    Charge un modèle sauvegardé.
    - Nouveau format : dict {sizes, biases, weights}
    - Ancien format (si tu as des vieux .pkl) : instance Network picklée
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)

    # Compatibilité avec les anciens fichiers où on picklait directement Network
    if isinstance(obj, Network):
        return obj

    # Nouveau format : dict de paramètres
    if isinstance(obj, dict):
        sizes = obj["sizes"]
        net = Network(sizes)
        net.biases = obj["biases"]
        net.weights = obj["weights"]
        return net

    raise TypeError(f"Format de modèle inconnu dans {path}: {type(obj)}")


def compute_confusion_matrix(net: Network, data, num_classes: int = 10):
    """
    Calcule une matrice de confusion (num_classes x num_classes)
    sur un dataset de la forme [(x, y_true), ...].

    Lignes  = classes réelles
    Colonnes = classes prédites
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for x, y_true in data:
        a = net.feedForward(x)
        y_pred = int(np.argmax(a))
        y_true = int(y_true)
        if 0 <= y_true < num_classes and 0 <= y_pred < num_classes:
            cm[y_true, y_pred] += 1

    return cm

# ==========================
#   SIDEBAR – CONTROLS
# ==========================

st.sidebar.header("Hyperparamètres")

epochs = st.sidebar.slider("Epochs", 1, 50, 10)
learning_rate = st.sidebar.slider("Learning rate (η)", 0.01, 5.0, 3.0, step=0.01)
mini_batch_size = st.sidebar.slider("Mini-batch size", 1, 100, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("Architecture")

hidden_size = st.sidebar.slider("Taille couche cachée", 10, 300, 100)
# Tu peux ajouter plusieurs couches plus tard (liste de sliders, etc.)

st.sidebar.markdown("---")
st.sidebar.subheader("Options d'entraînement")

use_validation = st.sidebar.checkbox("Utiliser validation comme test", value=False)
limit_train = st.sidebar.number_input(
    "Limiter le nb d'exemples d'entraînement (0 = tout)",
    min_value=0,
    max_value=len(training_data),
    value=0,
    step=1000,
)

# ==========================
#   LAYOUT PRINCIPAL
# ==========================

tab_readme, tab_train, tab_draw, tab_activations, tab_weights, tab_metrics, tab_errors = st.tabs(
    ["📖 Readme", "📡 Entraînement", "🖊️ Dessiner & Tester","✨ Activations", "🧮 Poids", "📈 Métriques", "🕵️ Erreurs"]
)

# ========== ONGLET README ==========
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




# ========== ONGLET TRAINING ==========

with tab_train:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Lancer un entraînement")

        start_training = st.button("🚀 Start training")

        st.markdown("### Terminal")

        # On crée (ou récupère) le placeholder pour le terminal
        if "log_placeholder" not in st.session_state or st.session_state.log_placeholder is None:
            st.session_state.log_placeholder = st.empty()

        log_box = st.session_state.log_placeholder

        # Affichage initial du contenu
        log_box.text_area(
            "Output",
            value=st.session_state.log_text,
            height=400,
        )

    with col_right:
        st.subheader("Dernier run")

        if st.session_state.current_run is not None:
            run = st.session_state.current_run
            st.write(f"**Run ID :** `{run['run_id']}`")
            st.write(f"**Date :** {run['timestamp']}")
            st.write(f"**Architecture :** {run['sizes']}")
            if "final_accuracy" in run:
                st.metric("Accuracy test", f"{run['final_accuracy']*100:.2f} %")
        else:
            st.info("Aucun run pour le moment.")

# ========== CALLBACK D'ENTRAÎNEMENT ==========

def make_epoch_callback(run_id):
    def epoch_callback(epoch, metrics, network: Network):
        # log text
        # if "test_accuracy" in metrics:
        #     append_log(
        #         f"[{run_id}] Epoch {epoch}/{metrics['epochs']} "
        #         f"- test_acc={metrics['test_accuracy']:.4f}"
        #     )
        # else:
        #     append_log(f"[{run_id}] Epoch {epoch}/{metrics['epochs']} complete")
        
        # ---------- Metrics history pour les graphes ----------
        entry = {
            "run_id": run_id,
            "epoch": epoch,
        }
        entry.update(metrics)
        st.session_state.metrics_history.append(entry)

        # ---------- Historique des poids de la première couche ----------
        if "weight_history" not in st.session_state:
            st.session_state.weight_history = {}

        if run_id not in st.session_state.weight_history:
            st.session_state.weight_history[run_id] = {
                "epochs": [],
                "w0_list": [],
            }

        hist = st.session_state.weight_history[run_id]

        # On logge l'epoch
        hist["epochs"].append(epoch)

        # Snapshot des poids de la première couche (input -> hidden)
        if len(network.weights) > 0:
            # copie pour ne pas être écrasé par les updates suivants
            hist["w0_list"].append(network.weights[0].copy())

    return epoch_callback
# ========== FONCTION POUR TRAINING AVEC PARAMÈTRES CUSTOM (AutoML) ==========

def run_single_training_with_params(eta, batch, cfg_name="automl"):
    """
    Identique à run_single_training(), mais accepte des hyperparamètres custom.
    Utilisé par AutoML pour éviter l'usage de global variables.
    """
    clear_log()

    # Prépare les données
    if limit_train and limit_train > 0:
        train_subset = training_data[:limit_train]
    else:
        train_subset = training_data

    test_set = validation_data if use_validation else test_data

    # Crée le réseau
    sizes = [784, hidden_size, 10]
    net = Network(sizes)

    run_id = f"{cfg_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    st.session_state.current_run = {
        "run_id": run_id,
        "sizes": sizes,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_name": cfg_name,
    }

    append_log(f"=== NEW RUN {run_id} ===")
    append_log(f"Architecture: {sizes}")
    append_log(f"Epochs={epochs}, eta={eta}, mini_batch={batch}")
    append_log(f"Train samples={len(train_subset)}, Test samples={len(test_set)}")
    append_log(f"======================================")

    # Lancer SGD avec callback
    net.SGD(
        training_data=train_subset,
        epochs=epochs,
        mini_batch_size=batch,
        eta=eta,
        test_data=test_set,
        log_fn=append_log,
        epoch_callback=make_epoch_callback(run_id),
    )

    # Accuracy finale
    correct = net.evaluate(test_set)
    final_acc = correct / len(test_set)
    st.session_state.current_run["final_accuracy"] = final_acc

    # Sauver modèle
    model_path = save_model(net, run_id, accuracy=final_acc)
    st.session_state.current_run["model_path"] = model_path

    # Pré-calcul erreurs
    st.session_state.misclassified_cache[run_id] = get_misclassified(net, test_set)

    return run_id, net

def run_single_training(cfg_name="manual"):
    """
    Entraînement normal en utilisant les valeurs des sliders
    (learning_rate, mini_batch_size).
    """
    return run_single_training_with_params(
        eta=learning_rate,
        batch=mini_batch_size,
        cfg_name=cfg_name,
    )

if start_training:
    run_single_training(cfg_name="manual")


# ========== ONGLET MÉTRIQUES ==========
with tab_metrics:
    st.subheader("📈 Métriques d'entraînement et de performance")

    if not st.session_state.metrics_history:
        st.info("Aucune métrique pour l'instant. Lance un entraînement pour voir les courbes.")
    else:
        import pandas as pd

        df = pd.DataFrame(st.session_state.metrics_history)
        run_ids = df["run_id"].unique().tolist()
        selected_run = st.selectbox("Sélectionne un run à analyser", run_ids)

        df_run = df[df["run_id"] == selected_run].sort_values("epoch")

        # ---------- COURBES DE BASE ----------
        st.markdown("### Courbes de base")

        col1, col2 = st.columns(2)

        with col1:
            if "test_accuracy" in df_run:
                st.line_chart(
                    df_run.set_index("epoch")["test_accuracy"],
                    height=300,
                )
                st.caption("Accuracy sur le set de test (ou validation) en fonction des epochs.")

        with col2:
            if "test_correct" in df_run:
                st.bar_chart(
                    df_run.set_index("epoch")["test_correct"],
                    height=300,
                )
                st.caption("Nombre de prédictions correctes par epoch.")

    # ---------- SUITE : GUIDES / VISUS AVANCÉES ----------
    st.markdown("---")
    st.markdown("### 📚 Guide de lecture des visualisations")

 # 1️⃣ Loss par epoch (avec courbe dans l'expander)
    with st.expander("🧠 Loss par epoch"):
        st.markdown(
            """
La **loss** mesure à quel point le modèle se trompe en moyenne.

- Après chaque epoch, on calcule une valeur de loss sur le jeu d'entraînement.
- Normalement, la loss doit **descendre** progressivement si le modèle apprend correctement.
"""
        )

        # On vérifie que metrics_history n'est pas vide
        if not st.session_state.metrics_history:
            st.info("Aucune loss disponible : lance un entraînement pour voir la courbe.")
        else:
            import pandas as pd

            df = pd.DataFrame(st.session_state.metrics_history)
            run_ids = df["run_id"].unique().tolist()
            
            selected_run_loss = st.selectbox(
                "Sélectionne un run pour afficher la loss",
                run_ids,
                key="select_run_loss"
            )

            df_run_loss = df[df["run_id"] == selected_run_loss].sort_values("epoch")

            # Vérification de la dispo de la loss
            if "train_loss" not in df_run_loss:
                st.warning("Ce run ne contient pas de valeurs de loss.")
            else:
                st.line_chart(
                    df_run_loss.set_index("epoch")["train_loss"],
                    height=300
                )

                last_loss = df_run_loss["train_loss"].iloc[-1]
                st.metric(
                    "Dernière loss enregistrée",
                    f"{last_loss:.4f}"
                )

        st.caption("La loss est calculée avec un MSE simple : 0.5 * || prédiction - vérité ||²")

    # 2️⃣ TSNE / PCA des embeddings de la hidden layer (interactif)
    with st.expander("🧩 TSNE / PCA de la couche cachée (interactif)"):
        st.markdown(
            """
On projette ici les activations de la **couche cachée** dans un plan 2D
pour voir comment le réseau sépare les chiffres dans son espace interne.

Chaque point = une image MNIST, colorée selon son vrai chiffre.
"""
        )

        # On ne peut rien faire tant qu'aucun modèle n'a été entraîné/sauvegardé
        if st.session_state.current_run is None or "model_path" not in st.session_state.current_run:
            st.info("Lance au moins un entraînement pour pouvoir calculer la projection TSNE/PCA.")
        else:
            run = st.session_state.current_run
            net = load_model(run["model_path"])

            import pandas as pd
            import altair as alt

            max_samples = len(test_data)
            if max_samples == 0:
                st.warning("Le set de test est vide, impossible de calculer la projection.")
            else:
                st.markdown("#### Paramètres de la projection")

                n_samples = st.slider(
                    "Nombre d'images à projeter",
                    min_value=100,
                    max_value=min(2000, max_samples),
                    value=min(500, max_samples),
                    step=100,
                    help="Plus il y a de points, plus la projection est riche, mais plus le calcul est long."
                )

                method = st.radio(
                    "Méthode de réduction de dimension",
                    ["PCA (rapide)", "t-SNE (plus lent, plus joli)"],
                    help="PCA donne une idée rapide, t-SNE donne souvent des clusters plus nets."
                )

                if st.button("Calculer la projection 2D"):
                    from sklearn.decomposition import PCA
                    from sklearn.manifold import TSNE

                    # Récupération des activations de la couche cachée
                    X = []
                    y_labels = []
                    for i, (x, y_true) in enumerate(test_data[:n_samples]):
                        _, activations = forward_with_activations(net, x)
                        hidden = activations[1]  # première couche cachée
                        X.append(hidden.ravel())
                        y_labels.append(int(y_true))

                    X = np.array(X)

                    # Choix du réducteur de dimension
                    if method.startswith("PCA"):
                        reducer = PCA(n_components=2)
                    else:
                        # t-SNE : plus lent, mais meilleure séparation visuelle
                        reducer = TSNE(
                            n_components=2,
                            init="random",
                            learning_rate="auto",
                            perplexity=min(30, n_samples - 1),
                        )

                    with st.spinner("Calcul de la projection en 2D..."):
                        emb = reducer.fit_transform(X)

                    df_emb = pd.DataFrame({
                        "x": emb[:, 0],
                        "y": emb[:, 1],
                        "label": y_labels,
                    })

                    st.markdown("#### Projection des embeddings de la couche cachée")

                    chart = alt.Chart(df_emb).mark_circle(size=50, opacity=0.8).encode(
                        x="x",
                        y="y",
                        color="label:N",
                        tooltip=["label:N"],
                    ).properties(
                        height=400
                    )

                    st.altair_chart(chart, use_container_width=True)

                    st.caption(
                        "Chaque point est une image MNIST projetée dans l'espace latent. "
                        "Les couleurs correspondent aux chiffres réels (0–9). "
                        "On cherche à voir si les classes se regroupent bien."
                    )

    # 4️⃣ Animation de l’évolution des poids d’un neurone
    with st.expander("🎞️ Animation de l’évolution des poids d’un neurone"):
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

        # --- Partie interactive : slider sur neurone & epoch ---
        if "weight_history" not in st.session_state or not st.session_state.weight_history:
            st.info("Aucun historique de poids disponible. Lance un entraînement pour commencer à enregistrer les poids.")
        else:
            import pandas as pd  # au cas où tu en as besoin plus bas

            run_ids_hist = list(st.session_state.weight_history.keys())

            selected_run_anim = st.selectbox(
                "Sélectionne un run pour visualiser l'évolution d'un neurone",
                run_ids_hist,
                key="select_run_weight_anim",
            )

            hist = st.session_state.weight_history.get(selected_run_anim, None)

            if hist is None or len(hist.get("w0_list", [])) == 0:
                st.warning("Pas encore d'historique de poids pour ce run.")
            else:
                epochs_hist = hist["epochs"]
                w0_list = hist["w0_list"]  # liste de matrices (hidden_size, 784)

                # On suppose que la taille de la couche cachée ne change pas au cours du run
                hidden_size_hist = w0_list[0].shape[0]

                col_sel1, col_sel2 = st.columns(2)
                with col_sel1:
                    neuron_idx = st.slider(
                        "Indice du neurone caché",
                        min_value=0,
                        max_value=hidden_size_hist - 1,
                        value=0,
                        key="anim_neuron_idx",
                    )
                with col_sel2:
                    if len(epochs_hist) <= 1:
                        # Un seul epoch disponible → pas de slider
                        epoch_pos = 0
                        st.info("Une seule epoch enregistrée pour ce run.")
                    else:
                        epoch_pos = st.slider(
                            "Epoch",
                            min_value=0,
                            max_value=len(epochs_hist) - 1,
                            value=len(epochs_hist) - 1,
                            key="anim_epoch_idx",
                        )

                epoch_val = epochs_hist[epoch_pos]
                # Poids du neurone sélectionné à cette epoch
                w_vec = w0_list[epoch_pos][neuron_idx, :]  # shape (784,)
                img = w_vec.reshape(28, 28)

                # Normalisation locale pour l'affichage
                w_min, w_max = img.min(), img.max()
                if w_max > w_min:
                    img_norm = (img - w_min) / (w_max - w_min)
                else:
                    img_norm = np.zeros_like(img)

                st.image(
                    img_norm,
                    width=160,
                    clamp=True,
                    caption=f"Run {selected_run_anim} – neurone {neuron_idx}, epoch {epoch_val}",
                )

                # Option : courbe de la norme des poids de ce neurone au cours du temps
                show_norms = st.checkbox(
                    "Afficher l'évolution de la norme des poids de ce neurone",
                    value=False,
                    key="show_neuron_norm_curve",
                )

                if show_norms:
                    norms = [float(np.linalg.norm(w0[neuron_idx, :])) for w0 in w0_list]
                    df_norm = pd.DataFrame(
                        {"epoch": epochs_hist, "weight_norm": norms}
                    ).set_index("epoch")
                    st.line_chart(df_norm)
                    st.caption("La norme des poids donne une idée de la 'force' du filtre appris par ce neurone.")

    # 5️⃣ Matrice de confusion
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

        # Vérifier qu'on a bien un modèle entraîné
        if st.session_state.current_run is None or "model_path" not in st.session_state.current_run:
            st.info("Lance un entraînement pour pouvoir calculer la matrice de confusion.")
        else:
            run = st.session_state.current_run
            net = load_model(run["model_path"])

            import pandas as pd
            import altair as alt

            # Choix du dataset
            dataset_choice = st.radio(
                "Dataset utilisé pour la matrice de confusion",
                ["Test set", "Validation set"],
                horizontal=True,
                key="confusion_dataset_choice",
            )

            if dataset_choice == "Test set":
                data = test_data
            else:
                data = validation_data

            if len(data) == 0:
                st.warning("Le dataset sélectionné est vide, impossible de calculer la matrice de confusion.")
            else:
                # Option : limiter le nombre d'exemples pour aller plus vite
                max_samples = len(data)
                n_samples = st.slider(
                    "Nombre d'images utilisées pour la matrice",
                    min_value=100,
                    max_value=max_samples,
                    value=min(1000, max_samples),
                    step=100,
                    key="confusion_n_samples",
                    help="Plus il y a d'images, plus la matrice est représentative (mais plus c'est long).",
                )

                if st.button("Calculer la matrice de confusion", key="confusion_button"):
                    subset = data[:n_samples]

                    with st.spinner("Calcul en cours..."):
                        cm = compute_confusion_matrix(net, subset, num_classes=10)

                    total = cm.sum()
                    correct = np.trace(cm)
                    acc = correct / total if total > 0 else 0.0

                    st.markdown(f"**Accuracy sur cet échantillon : {acc*100:.2f} %**")

                    # Préparer les données pour un heatmap Altair
                    df_cm = pd.DataFrame(cm, index=range(10), columns=range(10))
                    df_plot = (
                        df_cm
                        .reset_index()
                        .melt(id_vars="index", var_name="pred", value_name="count")
                        .rename(columns={"index": "true"})
                    )

                    st.markdown("### Heatmap de la matrice de confusion")

                    chart = (
                        alt.Chart(df_plot)
                        .mark_rect()
                        .encode(
                            x=alt.X("pred:O", title="Classe prédite"),
                            y=alt.Y("true:O", title="Classe réelle"),
                            color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
                            tooltip=["true", "pred", "count"],
                        )
                        .properties(height=400)
                    )

                    st.altair_chart(chart, use_container_width=True)

                    st.caption(
                        "Les valeurs sur la diagonale correspondent aux prédictions correctes. "
                        "Les cases hors diagonale montrent quelles classes sont le plus souvent confondues."
                    )


# ========== ONGLET ERREURS ==========

with tab_errors:
    st.subheader("Images mal classées")

    if st.session_state.current_run is None:
        st.info("Lance un run pour voir les erreurs.")
    else:
        run_id = st.session_state.current_run["run_id"]
        miscls = st.session_state.misclassified_cache.get(run_id)

        if not miscls:
            st.info("Pas d'erreurs trouvées (ou pas encore calculées).")
        else:
            st.write(f"{len(miscls)} exemples mal classés (montrés au max).")

            cols = st.columns(8)
            for i, (x, y_true, y_pred, probs) in enumerate(miscls):
                col = cols[i % len(cols)]
                img = np.reshape(x, (28, 28))
                with col:
                    st.image(img, width=60, caption=f"True:{y_true} / Pred:{y_pred}")

# ========== ONGLET ACTIVATIONS ==========

with tab_activations:
    st.subheader("Explorateur d'activations")

    if st.session_state.current_run is None or "model_path" not in st.session_state.current_run:
        st.info("Lance un entraînement pour analyser les activations.")
    else:
        run = st.session_state.current_run
        net = load_model(run["model_path"])

        index = st.slider(
            "Index d'image dans le set de test",
            min_value=0,
            max_value=len(test_data) - 1,
            value=0,
        )

        x, y_true = test_data[index]
        img = np.reshape(x, (28, 28))

        col_img, col_info = st.columns([1, 2])

        with col_img:
            st.image(img, width=140, caption=f"Label vrai : {int(y_true)}")

        with col_info:
            zs, activations = forward_with_activations(net, x)
            output = activations[-1]
            probs = softmax(output)

            st.markdown("**Distribution des sorties (softmax)**")
            import pandas as pd
            df_probs = pd.DataFrame({
                "digit": list(range(10)),
                "proba": probs.ravel(),
            })
            st.bar_chart(df_probs.set_index("digit"))

            st.markdown("**Normes d'activation par couche**")
            norms = [float(np.linalg.norm(a)) for a in activations]
            df_norms = pd.DataFrame({
                "layer": list(range(len(norms))),
                "activation_norm": norms,
            })
            st.line_chart(df_norms.set_index("layer"))

# ========== ONGLET POIDS ==========

with tab_weights:
    st.subheader("Analyse des poids")

    if st.session_state.current_run is None or "model_path" not in st.session_state.current_run:
        st.info("Lance un entraînement pour voir les poids.")
    else:
        run = st.session_state.current_run
        net = load_model(run["model_path"])

        stats = compute_weight_stats(net)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean", f"{stats['mean']:.4e}")
        col2.metric("Std", f"{stats['std']:.4e}")
        col3.metric("Min", f"{stats['min']:.4e}")
        col4.metric("Max", f"{stats['max']:.4e}")

        st.markdown("---")
        st.markdown("### Représentation des poids entre la couche d’entrée et la couche cachée")

        
        w0 = net.weights[0]  # shape (hidden_size, 784)
        cols = st.columns(10)
        for i in range(min(10, w0.shape[0])):
            col = cols[i % len(cols)]
            with col:
                img_w = w0[i, :].reshape(28, 28)

                # Normalisation locale pour ce neurone : [0, 1]
                w_min = img_w.min()
                w_max = img_w.max()
                if w_max > w_min:
                    img_w_norm = (img_w - w_min) / (w_max - w_min)
                else:
                    # cas dégénéré : tous les poids identiques
                    img_w_norm = np.zeros_like(img_w)

                st.image(img_w_norm, width=60, caption=f"Neuron {i}")


        st.markdown("---")
        st.markdown("### Représentation des poids entre la couche cachée et la couche de sortie")

        # Les poids de la dernière couche : shape (10, hidden_size)
        w_out = net.weights[-1]  

        # Chaque neurone de sortie utilise les 'features' produites par les hidden neurons
        # On va reconstruire une image 28x28 en faisant une combinaison pondérée 
        # des poids input->hidden, pondérée par les poids hidden->output.

        w_hidden = net.weights[0]   # shape (hidden_size, 784)

        cols = st.columns(10)
        for digit in range(10):
            col = cols[digit % len(cols)]
            with col:
                # Combinaison linéaire des filtres cachés
                # w_out[digit]: shape (hidden_size,)
                combined = np.dot(w_out[digit], w_hidden)  # shape (784,)

                # reshape en image
                img = combined.reshape(28, 28)

                # Normalisation locale pour l'affichage
                mn, mx = img.min(), img.max()
                if mx > mn:
                    img_norm = (img - mn) / (mx - mn)
                else:
                    img_norm = np.zeros_like(img)

                st.image(img_norm, width=60, caption=f"Classe {digit}")

        
        st.markdown("---")
        st.markdown("### Histogramme global des poids")

        all_weights = np.concatenate([w.ravel() for w in net.weights])
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.hist(all_weights, bins=50)
        ax.set_title("Distribution des poids")
        ax.set_xlabel("valeur")
        ax.set_ylabel("fréquence")
        st.pyplot(fig)

# ========== ONGLET : Dessiner & Tester ==========

with tab_draw:
    st.subheader("🖊️ Dessine un chiffre et teste un modèle MNIST")

    st.markdown("""
    Dessine un chiffre (0–9) dans la zone ci-dessous.  
    Il sera automatiquement **converti en MNIST 28×28** puis envoyé au modèle.
    """)

    # Pour pouvoir reset le canvas en changeant la key
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0

    # 1. Choix du modèle sauvegardé
    saved_files = list(st.session_state.saved_models.values())

    if not saved_files:
        st.warning("Aucun modèle sauvegardé. Entraîne un réseau pour en générer un.")
    else:
        model_file = st.selectbox("Choisis un modèle :", saved_files)
        net = load_model(model_file)

        st.markdown("### Zone de dessin")

        # Bouton pour effacer le canvas
        if st.button("🧽 Effacer le dessin"):
            st.session_state.canvas_key += 1  # change la key pour reset le canvas

        # Canvas Streamlit (key dépendante de canvas_key)
        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=400,
            width=400,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
        )

        # Bouton de prédiction
        if st.button("🔍 Prédire le chiffre dessiné"):
            if canvas_result.image_data is None:
                st.error("Dessin vide (ou pas encore de trait détecté).")
            else:
                img = canvas_result.image_data

                # Convertir en niveaux de gris
                img_gray = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_BGR2GRAY)

                # Inverser les couleurs (MNIST = fond noir, écriture claire)
                #img_gray = cv2.bitwise_not(img_gray)

                # Redimensionner en 28×28
                img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)

                # Normaliser 0–1
                img_norm = img_resized / 255.0

                # Aplatir en vecteur (784, 1)
                x = img_norm.reshape(784, 1)

                # Prédiction
                output = net.feedForward(x)
                probs = softmax(output)
                prediction = int(np.argmax(probs))

                st.markdown("### 📌 Résultat")
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.image(img_resized, width=150, caption="Image 28×28 envoyée au modèle")
                with col_b:
                    st.success(f"**Le modèle prédit : {prediction}**")
                    st.bar_chart(probs.ravel())