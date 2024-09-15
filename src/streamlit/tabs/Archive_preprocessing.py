import streamlit as st
import os
from functions.processing_functions import get_primitive_elts, store_dataframe_in_context, text_pruning, clean_text, generate_wordcloud_from_dataframe, remove_duplicated_from_project_dataset, selected_size_picking

title = "Traitement préliminaire des données"
sidebar_name = "Traitement préliminaire des données"

def run():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(dir_path, "../../assets/images/eda_advanced.png")
    st.image(img_path)

    st.title(title)

    st.markdown("---")

    st.markdown("Dans cette partie, nous poursuivons l'analyse exploratoire des données. Il est notamment question de l'étude du sens des posts et de faire ressortir les sujets saillants.")

    st.markdown("##### Attention, le traitement et le chargement des données peuvent prendre un certain temps.")

    st.write('### Analyse des mots "primitifs"')

    st.markdown("En procédant la suppression de caractères spéciaux comme la ponctuation et guillemets, nous avons pu obtenir une liste de mots primitifs. Voici quelques statistiques sur les mots primitifs :")

    with st.spinner("Traitement préliminaire des données en cours..."):
        get_primitive_elts(st.session_state.df)

    st.markdown("### Élagage des scories")

    st.markdown("La première des choses à faire est de supprimer les doublons évidents. En effet, il est possible que des posts soient dupliqués dans le jeu de données.")

    with st.spinner("Élagage des données en cours..."):
        pruned_df = text_pruning(st.session_state.df)

        st.markdown("L'étape suivante consiste à apppliquer un traitement de suppression des schémas particuliers comme les URL ou les courriels via l'utilisation d'expressions régulières.")

        pruned_df = clean_text(pruned_df)

        if pruned_df is not None:
            st.write("Voici les 5 premières lignes du jeu de données après élagage :")
            st.write(pruned_df.head(5))
        
    st.markdown("### Affichage du nuages de mots")

    st.markdown("Un ***word cloud*** (ou ***nuage de mots*** en français) est une représentation visuelle des mots contenus dans un texte, où la taille de chaque mot reflète sa fréquence ou son importance. Plus un mot apparaît souvent dans le texte, plus il est grand et visible dans le nuage.")

    range= [100, 50, 20]
    for i in range:
        generate_wordcloud_from_dataframe(pruned_df, 'post', i)

    st.markdown("### Analyse de la similarité des textes")

    st.markdown("Nous allons maintenant analyser la similarité des textes dans le jeu de données. Pour ce faire, nous allons utiliser la méthode ***TF-IDF*** (Term Frequency-Inverse Document Frequency) pour extraire les caractéristiques des textes et calculer la similarité entre eux.")
    
    st.markdown("###### Ce traitement peut prendre plusieurs minutes.")

    df_final = remove_duplicated_from_project_dataset(pruned_df)

    df_final = selected_size_picking(df_final)

    store_dataframe_in_context(df_final, key="df_final")

    if df_final is not None:
        st.markdown("Voici le résultat du traitement effecuté :")
        st.write(df_final.head(50))