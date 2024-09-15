import streamlit as st
import os

from functions.processing_functions import initialize_app

title = "Benchmark de LLMs pour la génération de résumés"
sidebar_name = "Introduction"

def run():
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(dir_path, "../../assets/images/benchmark_rect.png")
    st.image(img_path)

    st.title(title)

    st.markdown("---")

    st.write("### Contexte")
    st.write("Depuis le lancement d'agents conversationnels comme ChatGPT fin 2022, la popularité des larges modèles de langages (LLMs) n'a cessé de croître auprès du grand public. Ces modèles, entraînés sur des corpus de textes massifs, sont capables de générer des textes de qualité sur une grande variété de tâches. Cependant, leur utilisation pour la génération de résumés reste encore peu explorée.")
    st.markdown("Dans ce contexte, nous avons décidé de réaliser une évaluation comparative (benchmark) des LLMs pour la génération de résumés. Notre objectif est de comparer les performances de différents modèles sur une tâche de génération de résumés de textes plus ou moins longs.")
    st.markdown("Notre jeu de données ‘***twitter-misinformation***’ provient de ***Huggingface***, où il a été publié le 20 avril 2023 par l’utilisateur ***@roupenminassian***. Ce jeu de données nous a été communiqué par un tiers et avons choisi de l’utiliser à cause de notre intérêt commun pour l’actualité contemporaine des États-Unis.")
    st.markdown("Ce jeu de données regroupe des textes postés sur ***X*** (ex-Twitter) dont les sujets concernent l’actualité des États-Unis d'Amérique. Le titre indique que l’objectif principal du jeu de données est l'entraînement des modèles de machine learning pour détecter de la désinformation en ligne. Ceci ne constitue pas l'objectif de notre étude.")
            
    st.write("### Objectif")
    
    st.markdown("**Notre objectif est de comparer les performances de différents modèles sur une tâche de génération de résumés de textes plus ou moins longs.** Plus précisément, nous nous concentrons sur l'évaluation des performances des modèles de langages sur la tâche de génération de résumés sur deux types de corpus : taille courte (50 à 280 caractères) et taille moyenne (280 à 3000 caracctères).")
    st.markdown("L'étude est centrée sur des modèles SLMs (Small Language Models) qui ont moins de paramètres que les LLMs classiques. Nous cherchons à savoir si ces modèles plus légers peuvent rivaliser avec les modèles les plus larges sur la tâche de génération de résumés.")
    st.markdown("Pour ce faire, nous avons sélectionné un ensemble de modèles SLMs que nous allons entraîner et évaluer sur notre jeu de données :")

    img_path2 = os.path.join(dir_path, "../../assets/images/models.png")
    st.image(img_path2)

    # Récupération des données
    remote_repo = "roupenminassian/twitter-misinformation"
    local_file = "../../../data/raw/df_source.csv"
    with st.spinner("Chargement des données..."):
        initialize_app(False, remote_repo, local_file)