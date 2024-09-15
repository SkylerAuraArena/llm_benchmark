import streamlit as st
import os
from functions.vizualization_functions import calculate_word_statistics, analyze_text_length, plot_length_distributions, plot_fake_news_distribution, plot_violin_length_distribution, plot_target_data

title = "Analyse exploratoire préliminaire des données"
sidebar_name = "Analyse exploratoire préliminaire des données"

def run():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(dir_path, "../../assets/images/eda.png")
    st.image(img_path)

    st.title(title)

    st.markdown("---")

    st.markdown("Dans cette partie, nous présentons différentes analyses exploratoires des données effectuées sur le jeu de données. Ces analyses permettent de mieux comprendre la nature des données et de dégager des tendances ou des informations intéressantes.")

    st.write("### Statistiques descriptives")

    st.markdown("Le jeu de données est composé de deux colonnes : ***post*** et ***label***. La colonne ***post*** contient les posts et la colonne ***label*** indique si le post est une fausse information ou non. Voici un aperçu des données :")

    calculate_word_statistics(st.session_state.df)

    st.markdown("### Analyse de la longueur des textes")

    st.markdown("Nous avons analysé la longueur des textes pour mieux comprendre la distribution des données. Voici quelques statistiques sur la longueur des textes :")

    analyze_text_length(st.session_state.df)

    with st.spinner("Création des graphiques de répartition en cours..."):
        plot_length_distributions(st.session_state.df)

        st.markdown("La plupart des posts font moins de **5000** caractères. Cependant, certains posts sont très longs. En poussant plus loin l'analyse, on constate que les posts étiquetés comme étant de la désinformation sont bien plus longs. En outre, on constate une très forte concentration de tweet dans la tranche **100-149** caractères ce qui est logique étant donné que, par défaut, le nombre max de caractère est 144 sur X. On décide donc de fixer notre limite haute à **4738** caractères qui correspond à **96%** des post du dataframe.")

    st.markdown("### Analyse des posts vides et courts")

    plot_target_data(st.session_state.df)

    st.markdown("### Analyse de la distribution des fausses informations")

    st.markdown("Enfin, même si ce n'est pas au centre de notre étude, nous avons analysé la distribution des fausses informations dans le jeu de données. La distribution peut être consultée ci-après :")

    with st.spinner("Création des graphiques realtifs à la désinformation en cours..."):
        plot_fake_news_distribution(st.session_state.df)

        st.markdown("###### Le graphique suivant peut être un peu long à charger. Merci de patienter.")

        plot_violin_length_distribution(st.session_state.df)