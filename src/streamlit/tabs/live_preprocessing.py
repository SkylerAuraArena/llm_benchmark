import streamlit as st
import os
import pandas as pd

from functions.processing_functions import add_information_columns, add_type_and_modify_cluster_id, clean_text, create_randomized_clusters, evaluate_clusters, filter_and_process_clusters, generate_wordcloud_from_dataframe, get_primitive_elts, interactive_cluster_selection, make_Kmeans_clustering, remove_duplicated_from_custom_dataset, rename_and_drop_columns, store_dataframe_in_context, selected_size_picking, text_pruning, plot_umap_clusters, classify_texts_and_split, create_topic_clusters
from functions.vizualization_functions import analyze_text_length, calculate_word_statistics, plot_length_distributions, plot_target_data

title = "Démonstration de pré-traitement des données"
sidebar_name = "Démonstration de pré-traitement des données"

def run():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    st.title(title)

    st.markdown("---")

    st.markdown("Dans cette partie, vous pouvez utiliser la pipeline précédemment décrite sur vos propres données.")

    # Liste des valeurs de n_clusters prédéfinies
    # cluster_options = [3, 9, 15, 30, 66, 100]
    # Sélection du nombre de clusters pour le kmeans
    selected_clusters = interactive_cluster_selection()

    # On vérifie que des valeurs ont été sélectionnées avant d'exécuter le clustering
    if selected_clusters:
        # File uploader widget
        uploaded_file = st.file_uploader("Choisissez un fichier", type=["csv", "xlsx"])

        # Check if a file has been uploaded
        if uploaded_file is not None:
            # Display file details
            st.write("Nom du fichier:", uploaded_file.name)
            
            try:
                # Check the file extension and read the file accordingly
                if uploaded_file.name.endswith('.csv'):
                    # Read CSV file into a DataFrame
                    df = pd.read_csv(uploaded_file)
                    st.write("Contenu du fichier CSV :")
                elif uploaded_file.name.endswith('.xlsx'):
                    # Read Excel file into a DataFrame
                    df = pd.read_excel(uploaded_file)
                    st.write("Contenu du fichier Excel :")
                else:
                    st.error("Type de fichier non pris en charge.")
            except Exception as e:
                st.error(f"Erreur lors de la lecture du fichier : {e}")

            if df is not None:
                st.write(df)

            # Stockage du DataFrame initial pré-traité dans le contexte de session
            try:
                store_dataframe_in_context(df, key="df_custom")
                st.markdown("✅ **Les données ont été stockées dans le contexte de session avec succès.**")
            except Exception as e:
                st.markdown(f"❌ **Erreur lors du stockage des données dans le contexte de session :** `{e}`")

            # Prétraitement des données
            try:
                df_custom = rename_and_drop_columns(df, columns_to_rename={'Unnamed: 0': 'original_id', 'text': 'post', 'label': 'misinformation'}, columns_to_drop=[df.columns[0]])
                df_custom = add_information_columns(df_custom)
                st.markdown("✅ **Prétraitement des données réussi.**")
            except Exception as e:
                st.markdown(f"❌ **Erreur lors du prétraitement des données :** `{e}`")
                return
            
            calculate_word_statistics(df_custom)

            analyze_text_length(df_custom)

            with st.spinner("Création des graphiques de répartition en cours..."):
                plot_length_distributions(df_custom)

            plot_target_data(df_custom)

            with st.spinner("Traitement préliminaire des données en cours..."):
                get_primitive_elts(df_custom)

            with st.spinner("Élagage des données en cours..."):
                df_custom = text_pruning(df_custom)
                
                df_custom = clean_text(df_custom)

                if df_custom is not None:
                    st.write(df_custom.head(5))

            range= [100]
            for i in range:
                generate_wordcloud_from_dataframe(df_custom, 'post', i)

            st.markdown("###### Le traitement de la similarité peut prendre plusieurs minutes. Merci de patienter.")

            df_custom_final = remove_duplicated_from_custom_dataset(df_custom)

            df_custom_final = selected_size_picking(df_custom_final, custom_dataset=True)

            if df_custom_final is not None:
                st.write("Voici le résultat du traitement effecuté :")
                st.write(df_custom_final.head(50))

            # Suppression des colonnes inutiles
            df_custom_final = rename_and_drop_columns(df_custom_final, columns_to_rename=None, columns_to_drop=['misinformation', 'primitive_words', 'has_similarity', 'similar_with', 'similarity_group', 'similarity_score'])

            try:
                with st.spinner("Clusterisation en cours..."):
                    df_custom_final = make_Kmeans_clustering(df_custom_final, selected_clusters)
                    st.markdown("✅ **Clusterisation des données réalisée avec succès.**")
            except Exception as e:
                st.markdown(f"❌ **Erreur lors du traitement de la clusterisation :** `{e}`")

            st.write(df_custom_final.head(50))

            with st.spinner("Génération graphique des clusters en cours..."):
                plot_umap_clusters(df_custom_final)
                evaluate_clusters(df_custom_final, st.session_state.doc_emb)

            # Suppression des colonnes inutiles
            df_custom_final = rename_and_drop_columns(df_custom_final, columns_to_rename={'km_labels100': 'topic'}, columns_to_drop=['post_prepr', 'prepr_sign_count'])

            st.markdown(
            """
            Les textes vont à présent être séparés en fonction de leur taille : 
            - Short : 50 à 280 caractères.
            - Medium : 280 à 3000 caractères.
            """)
            st.markdown("Ceci nous permettra d'obtenir des regroupements de textes de tailles similaires à partir des clusters précédemment obtenus.")

            # Classification des textes et split en fonction de la taille
            df_custom_final, df_short, df_medium = classify_texts_and_split(df_custom_final)
            
            with st.spinner("Génération des regroupements aléatoires par taille de textes en cours..."):
                df_short_final = create_randomized_clusters(df_short)
                df_medium_final = create_randomized_clusters(df_medium)
                final_clusters_short = add_type_and_modify_cluster_id(df_short_final, 'short')
                final_clusters_medium = add_type_and_modify_cluster_id(df_medium_final, 'medium')

                # Concaténation des deux DataFrames
                randomized_clusters = pd.concat([final_clusters_medium, final_clusters_short], axis=0, ignore_index=True)
                st.markdown("✅ **Génération aléatoire réalisée avec succès.** Voici le résultat :")
                st.write(randomized_clusters.head(50))

                # Stockage du DataFrame final dans le contexte de session
                try:
                    store_dataframe_in_context(randomized_clusters, key="randomized_clusters")
                    # st.markdown("✅ **Les données ont été stockées dans le contexte de session avec succès.**")
                except Exception as e:
                    st.markdown(f"❌ **Erreur lors du stockage des données dans le contexte de session :** `{e}`")

            with st.spinner("Génération des regroupements pré-classifiés par taille de textes en cours..."):
                
                # Créer les clusters initiaux pour short et medium
                clusters_short = create_topic_clusters(df_short)
                clusters_medium = create_topic_clusters(df_medium)

                # Filtrer et traiter les clusters pour short et medium
                final_clusters_short = filter_and_process_clusters(clusters_short, df_short, cluster_type='short')
                final_clusters_medium = filter_and_process_clusters(clusters_medium, df_medium, cluster_type='medium')

                # Combiner les résultats finaux
                preclassified_clusters = pd.concat([final_clusters_medium, final_clusters_short], axis=0, ignore_index=True)
                preclassified_clusters['type'] = 'PC_' + preclassified_clusters['type']

                st.markdown("✅ **Génération pré-classifiée réalisée avec succès.** Voici le résultat :")
                st.write(preclassified_clusters.head(50))

                # Stockage du DataFrame final dans le contexte de session
                try:
                    store_dataframe_in_context(preclassified_clusters, key="preclassified_clusters")
                    st.markdown("✅ **Les données ont été stockées dans le contexte de session avec succès.**")
                except Exception as e:
                    st.markdown(f"❌ **Erreur lors du stockage des données dans le contexte de session :** `{e}`")
    else:
        st.warning("⚠️ Veuillez sélectionner au moins un nombre de clusters pour lancer la clusterisation.")