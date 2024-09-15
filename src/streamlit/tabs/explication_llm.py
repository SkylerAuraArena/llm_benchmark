import streamlit as st
import os

title = "Explication du système de génération des résumés"
sidebar_name = "Explication du système de génération des résumés"

def run():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_path0 = os.path.join(dir_path, "../../assets/images/summaries.png")
    st.image(img_path0)

    st.title(title)

    st.markdown("---")

    # img_path1 = os.path.join(dir_path, "../../assets/slides/Slides Soutenance DataScientest_page-0004.png")
    # st.image(img_path1)

    # img_path2 = os.path.join(dir_path, "../../assets/slides/Slides Soutenance DataScientest_page-0005.png")
    # st.image(img_path2)

    st.write("### Problématiques") 

    st.markdown("5 LLMs pour 1 jeu de 100 échantillons “random” et 1 jeu de 100 échantillons “préclassifiés” et 2 type de prompts.")

    st.markdown(
    """
        - ***Gestion du volume*** : très grand nombre d’appels LLMs 4000 Résumés (+ 800 résumés de référence) et 8000 Evaluations (Ref Free et Ref Based) ~13000 appels.
        - ***Gestion du prompt*** : syntaxe et précision, taille du prompt/contexte.
        - ***Gestion du temps*** : durée des appels, timeout, asynchronisme, batchs (utilisation GPU), particularités OpenAI (free tier).
        - ***Gestion des retours*** : hallucinations, ratio.
    """)

    st.write("### Réalisation") 

    st.markdown("2 Fonctions permettant de réaliser les résumés/évaluations avec paramètres :")
                
    st.markdown(
    """
        - sur une plage donnée,
        - par batch (optimisation de la charge GPU)
        - choix du/des LLMs
        - sur un model distant (RunPod) ou en local
        - activation ***OpenAI*** (résumés de référence)
        - choix du LLM juge (pour les évaluations)
        - appels asynchrones pour gérer le timeout en cas d’***hallucinations***
        - traçabilité,gestion des erreurs (enregistrements réguliers)
        - rate limit pour ***OpenAI***
    """)

    texts = [0, 4096, False]
    st.write(f"Utilisation de la librairie ***Langchain*** méthodes ***ChatOllama*** et ***ChatOpenAI***, température=", texts[0], ", num_ctx=", texts[1], " et stream=", texts[2], ".")

    st.write("##### Durées moyennes :") 

    st.markdown("***En local*** : RTX2070 (8Go) 10 à 30 sec par résumés et 5 à 10 pour les évaluations. : 22hr pour les résumés et 9hr évaluations.")
    st.markdown("***Sur RunPod*** : RTX A4500 (20Go) 5 sec par résumés et 5 sec pour les évaluations => 0,2€/hr.")
    st.markdown("***Via OpenAI*** : limite de 200 résumés par jour (4 jours au total) mais 0,5 à 1,5 sec par résumés => 0,0015€/résumé (1,2 € au total)")