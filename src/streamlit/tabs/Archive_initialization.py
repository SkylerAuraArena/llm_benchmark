import streamlit as st
import os

title = "Initialisation"
sidebar_name = "Initialisation"

def run():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(dir_path, "../../assets/images/init.png")
    st.image(img_path)

    st.title(title)

    st.markdown("---")

    st.markdown("Dans cette partie, nous allons voir quels sont les traitements préliminaires que nous avons effectués sur le jeu de données afin de les préparer pour une étude approfondie. Comme expliqué dans la description des données, nous avons récupérer de ***Huggingface*** deux bases de données :")
        
    st.markdown(
                """
                - tain.csv
                - test.csv""")
                
    st.markdown(
    """
    Le jeux de données comporte les colonnes suivantes : 
    - ***Unnamed: 0.1*** : l'index dans le jeu de données.
    - ***Unnamed: 0*** : il exactement du même index que la colonne précédente.
    - ***text*** : contenu du post X.
    - ***label*** : étiquette indiquant le classement éventuel en tant que désinformation (fake news). Il n'y a pas d'information accessible concernant la méthodologie d'étiquetage.
    """)

    st.write("### Concaténation") 

    st.markdown("Ces deux jeux de données ont été fusionnés afin d'obtenir un jeu de données unique nommé ***df***.")

    st.write("### Traitement des noms de colonnes et suppression de colonnes unitiles") 

    st.markdown("Nous avons effectué les opération suivantes sur notre jeu de données unique :")

    st.markdown(
                """
                - Suppression de la première colonne (identique à la deuxième colonne)
                - Deuxième colonne renommée ***original_id***
                - Troisième colonne ***text*** renommée en ***post***
                - Quatrième colonne ***label*** renommée en ***misinformation***""")
    
    st.write("### Ajout de colonnes d'informations") 

    st.markdown("Nous avons ensuite ajouté deux colonnes indiquant le nombre de caractères (***length***) et le nombre de mots (***words***) de chaque post.")   

    st.write("### Aperçu du jeu de données final")

    if st.session_state.df is not None:
        st.write(st.session_state.df.head(5))
        st.write(st.session_state.df.shape)