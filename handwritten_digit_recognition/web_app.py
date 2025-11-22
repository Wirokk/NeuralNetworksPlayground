import os
import pickle
from datetime import datetime

import numpy as np
import streamlit as st

import mnist_loader
from neural_network import Network

import cv2
from streamlit_drawable_canvas import st_canvas


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

# ==========================
#   CONFIG STREAMLIT
# ==========================

st.set_page_config(
    page_title="MNIST Lab – DL Playground",
    layout="wide"
)

st.title("🧠 MNIST Deep Learning Lab")
st.caption("Petit labo interactif pour explorer ton réseau neuronal MNIST.")

st.markdown(
    """
Bienvenue dans **ton mini-lab MLOps** :

- Configure le réseau et les hyperparamètres
- Lances l'entraînement
- Observe les **courbes de métriques**
- Inspecte les **erreurs**, les **activations**, les **poids**
- Teste un **mini AutoML** (recherche d'hyperparamètres)
"""
)

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
    os.makedirs("saved_models", exist_ok=True)
    path = os.path.join("saved_models", f"{run_id}_{accuracy:.4f}.pkl" if accuracy is not None else f"{run_id}.pkl")
    with open(path, "wb") as f:
        pickle.dump(net, f)
    st.session_state.saved_models[run_id] = path
    return path

def load_model(path: str) -> Network:
    with open(path, "rb") as f:
        net = pickle.load(f)
    return net

# ==========================
#   SIDEBAR – CONTROLS
# ==========================

st.sidebar.header("🎛 Hyperparamètres & Réseau")

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

tab_train, tab_metrics, tab_errors, tab_activations, tab_weights, tab_draw = st.tabs(
    ["📡 Entraînement", "📈 Métriques", "🕵️ Erreurs", "✨ Activations", "🧮 Poids", "🖊️ Dessiner & Tester"]
)

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
        if "test_accuracy" in metrics:
            append_log(
                f"[{run_id}] Epoch {epoch}/{metrics['epochs']} "
                f"- test_acc={metrics['test_accuracy']:.4f}"
            )
        else:
            append_log(f"[{run_id}] Epoch {epoch}/{metrics['epochs']} complete")

        # metrics history pour les graphes
        entry = {
            "run_id": run_id,
            "epoch": epoch,
        }
        entry.update(metrics)
        st.session_state.metrics_history.append(entry)
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
    append_log(f"Sizes: {sizes}")
    append_log(f"Epochs={epochs}, eta={eta}, mini_batch={batch}")
    append_log(f"Train samples={len(train_subset)}, Test samples={len(test_set)}")

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
    st.subheader("Courbes de métriques")

    if not st.session_state.metrics_history:
        st.info("Aucune métrique pour l'instant. Lance un entraînement.")
    else:
        import pandas as pd

        df = pd.DataFrame(st.session_state.metrics_history)
        run_ids = df["run_id"].unique().tolist()
        selected_run = st.selectbox("Sélectionne un run", run_ids)

        df_run = df[df["run_id"] == selected_run].sort_values("epoch")

        col1, col2 = st.columns(2)

        with col1:
            if "test_accuracy" in df_run:
                st.line_chart(
                    df_run.set_index("epoch")["test_accuracy"],
                    height=300,
                )
                st.caption("Accuracy test par epoch")
        with col2:
            if "test_correct" in df_run:
                st.bar_chart(
                    df_run.set_index("epoch")["test_correct"],
                    height=300,
                )
                st.caption("Nb de prédictions correctes par epoch")

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
        st.markdown("### Poids par classe (couche hidden → output)")

        show_output_weights = st.checkbox(
            "Afficher les poids de la couche de sortie (1 image par classe)",
            value=True
        )

        if show_output_weights:
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
        st.markdown("### Poids en entrée pour chaque classe (optionnel)")

        show_weight_images = st.checkbox("Afficher les poids comme images (couche input->hidden)", value=True)
        if show_weight_images and len(net.weights) > 0:
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