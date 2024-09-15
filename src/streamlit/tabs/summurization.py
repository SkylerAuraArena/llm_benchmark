import streamlit as st
import os

import pandas as pd
import asyncio

from functions.llm_functions import calculate_bert_scores, cosineSim_usingST, evaluateSummaries, getReferenceSummary, initialize_metrics, is_valid_url, interactive_model_selection, plot_interaction_charts, postprocess_dataframe, summarize, transpose_dataframe, validate_integer_input
from functions.processing_functions import store_dataframe_in_context
from sentence_transformers import SentenceTransformer

from params.prompts import short_prompt, elaborated_prompt_random, evaluation_prompts

title = "Réalisation et évaluation des résumés par les modèles de langage"
sidebar_name = "Réalisation et évaluation des résumés"

def run():
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(dir_path, "../../assets/images/llms.png")
    st.image(img_path)

    st.title(title)

    st.markdown("---")

    # Vérification de la présence des clusters sur lesquels effectuer les résumés
    try:
        # Vérifie si les clusters sont présents et non vides
        loaded_data = (
            st.session_state.randomized_clusters is not None and not st.session_state.randomized_clusters.empty and
            st.session_state.preclassified_clusters is not None and not st.session_state.preclassified_clusters.empty
        )
        # On vérifie que des valeurs ont été sélectionnées avant d'exécuter le clustering
        if loaded_data:
            st.markdown("Pour rappel, les modèles de langage sélectionnés pour les résumés sont les suivants :")
            img_path2 = os.path.join(dir_path, "../../assets/images/models.png")
            st.image(img_path2)
            
            # Sélection des modèles de langage
            selected_models = interactive_model_selection()
            # Indication du nombre de lignes à traiter
            number_of_posts = st.text_input("Veuillez indiquer le nombre de lignes à résumer :")
            # Validation de l'entrée de l'utilisateur (attente d'un nombre entier)
            validated_number = validate_integer_input(number_of_posts)
            # Entrée de l'utilisateur pour l'URL
            user_input = st.text_input("Veuillez indiquer l'adresse URL de votre serveur hébergeur des modèles de langage pour réaliser les résumés :")
            if user_input and selected_models and validated_number:
                if is_valid_url(user_input):
                    if st.button("Lancer la génération des résumés"):
                        # Récuparation des prompts
                        prompts_summaries={"short":short_prompt,"elaborate":elaborated_prompt_random}
                        with st.spinner("Génération des résumés en cours..."):
                            df = st.session_state.preclassified_clusters[:validated_number]
                            # Transposition du DataFrame pour les résumés
                            df = transpose_dataframe(df,selected_models,['cluster_id', 'clustered_text', 'id_50_3000', 'token_number', 'topic','type'])
                            # On réinitialise l'index du DataFrame
                            df.reset_index(drop=True, inplace=True)
                            # Génération des résumés
                            summaries = asyncio.run(summarize(df ,prompts=prompts_summaries,models=selected_models,start_idx=0,end_idx=validated_number,remote_url=user_input))
                            summaries['ref_summary'] = summaries.index.to_series().apply(lambda x: getReferenceSummary(summaries, x))
                            st.markdown("✅ **Génération des résumés réalisée avec succès.**")
                        
                        with st.spinner("Évaluation des résumés en cours..."):     
                            evaluateSummaries(summaries, evaluation_prompts, selected_models, eval_type=["RefFree", "RefBased"], start_id=0, end_idx=validated_number, judge_model="gpt3.5", remote_url=user_input, boostVariance=False, forceEvaluation=True)

                            summaries['Overall_llm'] = pd.to_numeric(summaries['Overall_llm'], errors='coerce')

                            summaries['refFree_llm_mean_score'] = summaries[['Clarity_llm', 'Accuracy_llm', 'Coverage_llm', 'Overall_llm']].mean(axis=1)
                            summaries['refBased_llm_mean_score'] = summaries[['Accuracy_llm', 'Conciseness_ref_llm', 'Structure_ref_llm']].mean(axis=1)

                            filter_mask = (summaries['model_name'] != 'gpt3.5') & (summaries["ratio"] <= 150) & (summaries['summary'] != 'Error timeout')
                            model = SentenceTransformer('all-MiniLM-L6-v2')
                            summaries.loc[filter_mask, 'cosine_sim_ref'] = summaries.loc[filter_mask].index.to_series().progress_apply(lambda idx: cosineSim_usingST(summaries, idx, model))
                            
                            st.markdown("✅ **Évaluation des résumés terminée.**")

                        with st.spinner("Génération des métriques en cours..."):     
                            # Utilisation de la fonction initialize_metrics pour le DataFrames summaries
                            summaries_with_metrics = initialize_metrics(summaries)                       
                            df_bert_scores = calculate_bert_scores(summaries_with_metrics)
                            st.markdown("✅ **Génération des métriques terminée.**")     

                            df_bert_scores = postprocess_dataframe(df_bert_scores)
                            st.write("Enfin, on affiche le jeu de données final :")
                            st.write(df_bert_scores)

                            st.write("Voici, à titre d'exemple, le résultat pour le premier résumé généré ainsi que son évaluation:")
                            st.write(df_bert_scores.iloc[0])

                            st.write("Ainsi que la métrique de ***BERT score*** par ***modèle*** en fonction du type de ***prompt*** :")
                            plot_interaction_charts(df_bert_scores)

                        # Stockage du DataFrame final dans le contexte de session
                        try:
                            store_dataframe_in_context(summaries, key="summuries")
                            st.markdown("✅ **Les résumés ont été stockés dans le contexte de session avec succès.**")
                        except Exception as e:
                            st.markdown(f"❌ **Erreur lors du stockage des résumés dans le contexte de session :** `{e}`")
                else:
                    st.error("❌ L'URL n'est pas valide. Veuillez vérifier le format.")
        else:
            st.markdown("⚠️ **Les clusters sont vides ou non chargés. Veuillez vérifier vos données.**")
    except Exception as e:
        try:
            if st.session_state.randomized_clusters is not None:
                st.markdown(f"❌ Un erreur s'est produite : {e}")
        except Exception as e:
            st.markdown(f"❌ **Attention, vous devez d'abord pré-traiter les données avant de réaliser des résumés**. Cette étape est accessible dans l'onglet **Démonstration de prétraitement des données**.")

    