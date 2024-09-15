# Import des bibliothèques standard
import os
import time
import json
import re
import asyncio

# Import des bibliothèques externes
import pandas as pd
import numpy as np
import streamlit as st
from tqdm import tqdm
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from textstat import textstat
from bert_score import BERTScorer

# Configuration de tqdm pour pandas
tqdm.pandas()

# Initialisation de asyncio lock
lock = asyncio.Lock()

# Import des modules spécifiques au projet
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from params.constants import full_models_list

# Chargement des variables d'environnement
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

def interactive_model_selection():
    """
    Permet à l'utilisateur de sélectionner de manière interactive les modèles à utiliser.
    """
    # Liste des valeurs de n_clusters prédéfinies
    cluster_options = full_models_list

    # Utiliser un multiselect pour permettre à l'utilisateur de choisir les valeurs
    selected_clusters = st.multiselect(
        'Sélectionnez les modèles à utiliser pour la génération de résumés :',
        options=cluster_options,
        default=[]  # Valeurs par défaut sélectionnées
    )

    # Retourner les valeurs sélectionnées pour les utiliser dans d'autres fonctions
    return selected_clusters

def is_valid_url(url):
    """
    Vérifie si l'URL correspond au format spécifique de RunPod.
    
    Args:
        url (str): L'URL à vérifier.
    
    Returns:
        bool: True si l'URL est valide, sinon False.
    """
    # Regex pour valider une URL du type spécifique runpod.net
    url_pattern = re.compile(
        r'^https://[a-z0-9]+-\d+\.proxy\.runpod\.net/$'
    )

    return bool(url_pattern.match(url))

def validate_integer_input(input_value):
    """
    Vérifie si l'entrée est un nombre entier. Affiche un message d'erreur si ce n'est pas le cas.

    Args:
        input_value (str): La valeur entrée par l'utilisateur.

    Returns:
        int or None: L'entier validé ou None si la validation échoue.
    """

    if input_value is not None and input_value != "":
        try:
            # Convertir l'entrée en entier
            validated_value = int(input_value)
            return validated_value
        except ValueError:
            # Afficher un message d'erreur si l'entrée n'est pas un nombre entier
            st.error("Veuillez entrer un nombre entier valide.")
            return None

async def call_llm(chain, doc, timeout=200):
    """
    Appelle un modèle de langage de manière asynchrone pour générer un résumé.

    Args:
        chain: La chaîne de traitement qui inclut le modèle de langage (LLM) et le prompt configuré.
        doc (str): Le document à résumer.
        timeout (int): Le temps maximum (en secondes) avant de déclencher une erreur de timeout.

    Returns:
        str ou object: La réponse du modèle si l'appel est réussi, ou un message d'erreur en cas d'échec.
    """
    try:
        # Utilisation de async_timeout pour limiter la durée de l'appel LLM
        async with async_timeout.timeout(timeout):
            # Appel asynchrone du modèle de langage avec le document en entrée
            response = await chain.ainvoke({"document": doc})
            return response

    except asyncio.TimeoutError:
        # Gestion du timeout : Affiche un message d'erreur et retourne un message spécifique
        print("Timeout!")
        # Retourne "Error timeout" pour signaler un échec dû à un dépassement de temps,
        # ce qui permet de passer au prochain résumé sans générer des résultats inattendus
        return "Error timeout"

    except Exception as e:
        # Gestion des autres erreurs : Affiche l'erreur et retourne un message d'erreur générique
        print(f"An error occurred: {e}")
        # Retourne "Error processing request" pour signaler un échec général de traitement
        return "Error processing request"
    
#----------------------------------------------------------

def initiate_chatgpt(force_reset=False):
    """
    Initialise la clé API pour ChatGPT en gérant l'ajout et la suppression de la clé dans les variables d'environnement.

    Args:
        force_reset (bool): Si True, supprime la clé API existante avant de demander une nouvelle entrée.
    """
    # Supprimer la clé API de l'environnement si force_reset est activé
    if force_reset:
        remove_api_key()

    # Vérifier si la clé API n'est pas déjà définie, sinon demander à l'utilisateur de l'entrer
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = request_api_key()

def remove_api_key():
    """
    Supprime la clé API d'OpenAI des variables d'environnement si elle est définie.
    """
    try:
        del os.environ["OPENAI_API_KEY"]
        print("API key removed from environment.")
    except KeyError:
        print("API key was not set.")

def request_api_key():
    """
    Demande à l'utilisateur d'entrer la clé API d'OpenAI via une entrée sécurisée.

    Returns:
        str: La clé API entrée par l'utilisateur.
    """
    #return getpass.getpass("Enter your OpenAI API key: ")
    return api_key

def gpt_price(response):
    """
    Calcule le coût total en fonction du nombre de tokens utilisés dans la réponse GPT.

    Args:
        response: Objet réponse du modèle GPT contenant des métadonnées sur l'utilisation des tokens.

    Returns:
        float: Le coût total de l'utilisation des tokens en dollars.
    """
    # Prix par million de tokens pour les entrées et les sorties
    input_token_price_per_million = 0.5
    output_token_price_per_million = 1.5

    # Conversion du prix par million de tokens en prix par token
    input_token_price_per_token = input_token_price_per_million / 1e6
    output_token_price_per_token = output_token_price_per_million / 1e6

    # Calcul du prix basé sur l'utilisation des tokens
    input_tokens = response.usage_metadata['input_tokens']
    output_tokens = response.usage_metadata['output_tokens']

    # Calcul du coût des tokens d'entrée et de sortie
    input_price = input_tokens * input_token_price_per_token
    output_price = output_tokens * output_token_price_per_token

    # Calcul du coût total
    total_price = input_price + output_price

    return total_price

#----------------------------------------------------------

def transpose_dataframe(df, models, columns_to_keep=None):
    """
    Transpose un DataFrame en dupliquant les lignes pour chaque combinaison de modèles et types de prompts,
    et en ajoutant des colonnes pour les résultats de résumé.

    Args:
        df (pd.DataFrame): Le DataFrame d'origine à transposer.
        models (list): Liste des modèles à inclure dans le DataFrame transposé.
        columns_to_keep (list, optional): Colonnes spécifiques à conserver dans le DataFrame transposé.

    Returns:
        pd.DataFrame: Le DataFrame transposé avec des colonnes pour les modèles, types de prompts, et résultats.
    """
    # Créer une copie du DataFrame avec un identifiant de base
    df_copy = df.copy()
    df_copy['base_id'] = df_copy.index

    # Définir les types de prompts
    prompt_types = ["Short", "Elaborate"]

    # Colonnes à conserver dans le DataFrame transposé
    columns = columns_to_keep if columns_to_keep else []
    columns.append('base_id')

    # Liste pour stocker les DataFrames intermédiaires
    dfs = []

    # Générer les combinaisons de modèles et types de prompts
    for model in models:
        for prompt_type in prompt_types:
            # Créer une copie des colonnes spécifiées et ajouter les colonnes de résultats
            df_temp = df_copy[columns].copy()
            df_temp['model_name'] = model
            df_temp['prompt_type'] = prompt_type
            df_temp[['summary', 'duration', 'ratio', 'price']] = np.nan  # Initialiser les colonnes de résultats à NaN
            dfs.append(df_temp)

    # Concaténer tous les DataFrames intermédiaires en un seul DataFrame
    df_transformed = pd.concat(dfs, ignore_index=True)

    # Ordonner les données par 'base_id', 'prompt_type' et 'model_name'
    df_transformed = df_transformed.sort_values(by=['base_id', 'prompt_type', 'model_name'])

    return df_transformed

#----------------------------------------------------------
# Les fonctions ci-après sont à optimiser pour être utilisées dans un contexte Streamlit
#----------------------------------------------------------

async def summarize(df,prompts,models=None,start_idx=0,end_idx=None,remote_url=None):

    if end_idx is None:
        end_idx = len(df)

    #Nombre maximal de token en input
    num_ctx=4096
    #Temperature pour les appels llms
    temperature=0
    #Temps pendant lequel on attend la réponse du llm, au delà on génère un Erreur timeout et on passe au suivant (ne marche pas pour le remote url)
    timeout=200

    last_execution_time = None
    call_count = 0
    limit_per_minute = 2000
    interval = 60.0 / limit_per_minute

    for i in tqdm(range(start_idx, end_idx, 1), desc="Processing posts"):
            current_row = df.iloc[i]
            #Get the model,prompt type
            current_model=df.loc[i,'model_name']
            current_prompt_type=df.loc[i,'prompt_type']
            current_doc=df.loc[i,'clustered_text']
            current_summary=df.loc[i,'summary']
            token_Initial=df.loc[i,'token_number']
            if models is None or (isinstance(models, list) and current_model in models) or current_model == models:
                #Le modèle courant est un des modèles sur lesquel on souhaite effectuer un résumé donc on poursuit
                if 'done' not in df.columns or (pd.isna(df.loc[i, 'done']) or df.loc[i, 'done'] != True):
                    #La colonne done n'existe pas ou si elle existe son contenu n'est pas renseigné ou n'est pas True, donc on fait des chose

                    #Selon le modèle courant on défini le llm utilisé
                    if current_model=="gpt3.5":
                        llm = ChatOpenAI(model="gpt-3.5-turbo-0125",streaming=False,temperature=temperature,max_tokens=num_ctx, api_key=api_key)
                    else:
                        if remote_url:
                            llm = ChatOllama(base_url=remote_url, model=current_model,streaming=False,safe_mode=False,verbose=True,temperature=temperature,num_ctx=num_ctx)
                        else:
                            llm = ChatOllama(model=current_model,streaming=False,safe_mode=False,verbose=True,temperature=temperature,num_ctx=num_ctx)
                    
                    #On défini la chaine d'appel
                    chain = prompts[current_prompt_type.lower()] | llm 

                    #On réalise l'appel au llm
                    try:
                        if current_model=="gpt3.5":
                            if call_count >= limit_per_minute:
                                time_elapsed = time.perf_counter() - last_execution_time
                                if time_elapsed < 60:
                                    print(f"Rate Limit for gpt3.5 turbo reached => sleep {60 - time_elapsed} seconds")
                                    time.sleep(60 - time_elapsed)
                                call_count = 0
                            
                            last_execution_time = time.perf_counter()
                            call_count += 1

                        #C'est ici qu'on réalise effectivement l'appel au llm en local, ou sur machine distante ou via openAI
                        start_time = time.perf_counter()
                        if remote_url:
                            #Lors d'un appel distant l'asynchronisme marche mal, on ne bénéficie donc pas de la gestion du timeout, ATTENTION il faut donc garder un oeil sur l'avancement....
                            # response = 'Error timeout'
                            response = chain.invoke({"document": current_doc})
                        else:
                            response = await call_llm(chain,current_doc,timeout)
                        end_time = time.perf_counter()
                        execution_time = end_time - start_time

                        async with lock:  # Acquire the lock before modifying the DataFrame
                            if isinstance(response, str) and response == "Error timeout":
                                error_message=f"Timeout error at : {i} for model {current_model} prompt type {current_prompt_type}"
                                print(error_message)
                                df.at[i, 'summary']="Error timeout"
                                df.at[i, 'error'] = error_message
                                df.at[i, 'done'] = True
                            elif isinstance(response, str) and response =="Error processing request":
                                error_message=f"Processing error at : {i} for model {current_model} prompt type {current_prompt_type}"
                                print(error_message)
                                df.at[i, 'error'] = error_message
                                df.at[i, 'done'] = False
                            else:
                                df.at[i, 'summary'] = response.content
                                if current_model!="gpt3.5":
                                    df.at[i, 'duration'] = response.response_metadata['total_duration']/1e9
                                    df.at[i, 'ratio'] = response.response_metadata['eval_count']/token_Initial*100
                                    df.at[i, 'input_tokens'] = response.response_metadata['prompt_eval_count']
                                else:
                                    df.at[i, 'duration'] = execution_time
                                    df.at[i, 'ratio'] = response.response_metadata['token_usage']['completion_tokens']/token_Initial*100
                                    df.at[i, 'input_tokens'] = response.usage_metadata['input_tokens']
                                    df.at[i, 'price'] = gpt_price(response)
                                df.at[i, 'done'] = True
                                df.at[i, 'error'] = np.nan
                    except Exception as e:
                        # Gestion des erreurs, sauvegarde dans un fichier et levée de l'exception
                        error_message = f"Error processing rows {i} : {e}"
                        print(error_message)
                        async with lock:
                            df.at[i, 'error'] = error_message
                            df.at[i, 'done'] = False
                else:
                    #La colonne done existe et sont contenu est True, donc on fait rien
                    continue
            else:
                #Le modèle courant n'est pas un des modèles retenu pour générer le résumé, on passe
                continue
    return df

#----------------------------------------------------------

def getReferenceSummary(df,id):
    if id not in df.index:
        raise ValueError("L'ID spécifié n'existe pas dans le DataFrame.")
    
    required_columns = ['cluster_id', 'prompt_type', 'model_name', 'summary']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"La colonne '{col}' est manquante dans le DataFrame.")
        
    # Filtrer le DataFrame pour trouver la ligne qui correspond au cluster_id, prompt_type, et model_name = "gpt3.5"
    filtered_df = df[
        (df['cluster_id'] == df.loc[id, 'cluster_id']) &
        (df['prompt_type'] == df.loc[id, 'prompt_type']) &
        (df['model_name'] == "gpt3.5")
    ]
    # Extraire le résumé
    summary = filtered_df['summary']
    # Gérer les cas où il y a plusieurs résumés ou aucun
    if len(summary) > 1:
        print("Attention : plusieurs résumés trouvés, retour du premier.")
        return summary.iloc[0]
    elif len(summary) == 0:
        print("Aucun résumé trouvé pour ces critères.")
        return None
    else:
        return summary.iloc[0]
    
def checkResult(response,label):
    value = response.get(label, np.nan)
    if isinstance(value, (int, float)):
        return value
    elif isinstance(value, str):
        try:
            return int(value)  # Essaie de convertir en entier
        except ValueError:
            try:
                return float(value)  # Si échec, essaie de convertir en float
            except ValueError:
                return np.nan  
    else:
        return np.nan
    
def evaluate_summary(df,prompts,models=None,eval_type=None,judge_model="gpt3.5",start_idx=0, end_idx=None,remote_url=None,boostVariance=False,forceEvaluation=False):

    if end_idx is None:
        end_idx = len(df)

    if judge_model=="gpt3.5":
        useGPT=True
    else:
        useGPT=False

    #Nombre maximal de token en input
    num_ctx=4096
    #Temperature pour les appels llms
    temperature=0
    #Temps pendant lequel on attend la réponse du llm, au delà on génère un Erreur timeout et on passe au suivant (ne marche pas pour le remote url)
    timeout=200

    refFree_counter=0
    refBased_counter=0
    refFree_error_counter=0
    refBased_error_counter=0
    refFree_already_counter=0
    refBased_already_counter=0
    

    for i in tqdm(range(start_idx, end_idx, 1), desc="Processing posts"):

        current_row = df.iloc[i]
        #Get the model,prompt type
        current_model=df.loc[i,'model_name']
        if current_model=="gpt3.5":
            #On ne fait aucune evaluation des résumés de référence
            continue
        
        current_prompt_type=df.loc[i,'prompt_type']
        current_doc=df.loc[i,'clustered_text']
        current_summary=df.loc[i,'summary']

        if(current_summary=="Error timeout" or current_summary=="Error ratio"):
            #Si le résumé n'a pas pu être calculé on ne fait aucune évaluation
            continue
        
        if models is None or (isinstance(models, list) and current_model in models) or current_model == models:
            #Le modèle courant est un des modèles sur lesquels on souhaite effectuer uen évaluation donc on poursuit

            #Selon le modèle juge on défini le llm utilisé
            if judge_model=="gpt3.5":
                llm = ChatOpenAI(model="gpt-3.5-turbo-0125",streaming=False,temperature=temperature,max_tokens=num_ctx, api_key=api_key)
            else:
                if remote_url:
                    llm = ChatOllama(base_url=remote_url, model=judge_model,streaming=False,safe_mode=False,verbose=True,temperature=temperature,num_ctx=num_ctx,format="json")
                else:
                    llm = ChatOllama(model=judge_model,streaming=False,safe_mode=False,verbose=True,temperature=temperature,num_ctx=num_ctx,format="json")

            #Ref Free Evaluation
            if eval_type is None or (isinstance(eval_type, list) and "RefFree" in eval_type) or eval_type=="RefFree":
                if forceEvaluation or ('done_refFree' not in df.columns or (pd.isna(df.loc[i, 'done_refFree']) or df.loc[i, 'done_refFree'] != True)):
                    
                    #On défini la chaine d'appel
                    if boostVariance:
                        prompt=prompts['refFree']['prompt_variance_boost']
                    else:
                        prompt=prompts['refFree']['prompt_standard']

                    chain = prompt | llm | JsonOutputParser()

                    try:
                        response_refFree = chain.invoke({"document": current_doc, "summary": current_summary, "example_output": prompts['refFree']['example_output']})
                        # print(response_refFree)
                        #Complete Dataframe
                        df.at[i, 'Clarity_llm'] = checkResult(response_refFree,'Clarity')
                        df.at[i, 'Accuracy_llm'] = checkResult(response_refFree,'Accuracy')
                        df.at[i, 'Coverage_llm'] = checkResult(response_refFree,'Coverage')
                        df.at[i, 'Overall_llm'] = checkResult(response_refFree,'Overall quality') 
                        df.at[i, 'Explanations_refFree_llm'] = json.dumps(response_refFree.get('Explanations','No explanation'))
                        df.at[i, 'error_refFree'] = np.nan
                        df.at[i, 'done_refFree'] = True
                        refFree_counter+=1
                    except Exception as e:
                        # Gestion des erreurs, sauvegarde dans un fichier et levée de l'exception
                        error_message = f"Error evaluating ref free at row {i} : {e}"
                        print(error_message)
                        df.at[i, 'error_refFree'] = error_message
                        df.at[i, 'done_refFree'] = False
                        refFree_error_counter+=1
                else:
                    refFree_already_counter+=1

            #Ref Based Evaluation
            if eval_type is None or (isinstance(eval_type, list) and "RefBased" in eval_type) or eval_type=="RefBased":
                if forceEvaluation or ('done_refBased' not in df.columns or (pd.isna(df.loc[i, 'done_refBased']) or df.loc[i, 'done_refBased'] != True)):
                    #On récupère le résumé de référence
                    ref_summary=df.loc[i,'ref_summary']
                    
                    if ref_summary=="" or pd.isna(ref_summary):
                        print(f"No reference summary for row {i}")
                    else:
                        #On défini la chaine d'appel
                        if boostVariance:
                            prompt=prompts['refBased']['prompt_variance_boost']
                        else:
                            prompt=prompts['refBased']['prompt_standard']

                        chain = prompt | llm | JsonOutputParser()

                        try:
                            response_refBased = chain.invoke({"ref_summary": ref_summary, "summary": current_summary,"example_output": prompts['refBased']['example_output']})
                            #Complete Dataframe
                            df.at[i, 'Accuracy_ref_llm'] = checkResult(response_refBased,'Accuracy')
                            df.at[i, 'Conciseness_ref_llm'] = checkResult(response_refBased,'Conciseness')
                            df.at[i, 'Structure_ref_llm'] = checkResult(response_refBased,'Structure') 
                            df.at[i, 'Explanations_refBased_llm'] = json.dumps(response_refBased.get('Explanations','No explanation'))
                            df.at[i, 'error_refBased'] = np.nan
                            df.at[i, 'done_refBased'] = True
                            refBased_counter+=1
                        except Exception as e:
                            # Gestion des erreurs, sauvegarde dans un fichier et levée de l'exception
                            error_message = f"Error evaluating ref based at row {i} : {e}"
                            print(error_message)
                            df.at[i, 'error_refBased'] = error_message
                            df.at[i, 'done_refBased'] = False
                            refBased_error_counter+=1
                else:
                    refBased_already_counter+=1

    st.write(f"Sur les ", refFree_counter, " évaluation(s) ref Free et ", refBased_counter, " évaluation(s) ref Based ont été générées, il y a eu ", refFree_error_counter, " erreur(s) ref Free et ", refBased_error_counter, " erreurs ref Based. À noter que ", refFree_already_counter, " évaluation(s) ref Free et ", refBased_already_counter, " évaluations ref Based avaient déjà été traitées.")
    
def evaluateSummaries(df,prompts,models=None,eval_type=None,start_id=0,end_idx=None,judge_model="gpt3.5",remote_url=None,boostVariance=False,forceEvaluation=False):
    if end_idx is None:
        end_idx = len(df)
    eval_types=[]
    if eval_type is None :
        eval_types=["RefFree","RefBased"]
    elif isinstance(eval_type, list):
        eval_types=eval_type
    else:
        eval_types.append(eval_type)

    evaluate_summary(df,prompts=prompts,models=models,eval_type=eval_types,judge_model=judge_model,start_idx=start_id,end_idx=end_idx,remote_url=remote_url,boostVariance=boostVariance,forceEvaluation=forceEvaluation)

def cosineSim_usingST(df,id_test, model_st, labelColumn="summary"):
    summary_test=df.loc[id_test,labelColumn]
    summary_test_ref= df.loc[id_test,'ref_summary']
    embeddings = model_st.encode([summary_test, summary_test_ref])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
    # print(f"Similarité cosinus: {similarity[0][0]:.4f}")
    return similarity[0][0]

#----------------------------------------------------------

def clean_and_filter_dataframe(df):
    """
    Nettoie et filtre le DataFrame en supprimant les lignes indésirables.

    Args:
        df (pd.DataFrame): Le DataFrame à nettoyer et filtrer.

    Returns:
        pd.DataFrame: Le DataFrame nettoyé et filtré.
    """
    # Vérifier et convertir 'ratio' en chaîne de caractères, puis en float
    df['ratio'] = pd.to_numeric(df['ratio'].astype(str).str.replace('%', ''), errors='coerce')

    # Filtrer les lignes selon les critères spécifiés
    filter_mask = (
        (df['model_name'] != "gpt3.5") &
        (df['ratio'] <= 150) &
        (df['duration'] <= 200) &
        (df['summary'] != "Error timeout") &
        (df['error'].isnull())
    )

    return df[filter_mask]

def calculate_metrics(df, scorer):
    """
    Calcule les métriques ROUGE, BLEU, Flesch Reading Ease et Dale-Chall pour chaque résumé du DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les résumés et résumés de référence.
        scorer (RougeScorer): Instance de RougeScorer pour calculer les scores ROUGE.

    Returns:
        pd.DataFrame: Le DataFrame enrichi avec les métriques calculées.
    """
    def compute_metrics(row):
        """
        Calcule les métriques pour une ligne donnée du DataFrame.

        Args:
            row (pd.Series): Une ligne du DataFrame.

        Returns:
            dict: Dictionnaire des scores calculés pour la ligne.
        """
        summary = str(row['summary']) if pd.notnull(row['summary']) else ""
        reference = str(row['ref_summary']) if pd.notnull(row['ref_summary']) else ""

        # Calcul des scores ROUGE
        rouge_scores = scorer.score(summary, reference)
        rouge1 = rouge_scores['rouge1'].fmeasure
        rouge2 = rouge_scores['rouge2'].fmeasure
        rougeL = rouge_scores['rougeL'].fmeasure

        # Calcul du score BLEU avec un lissage
        reference_tokens = [reference.split()]
        summary_tokens = summary.split()
        smoothing = SmoothingFunction().method4

        try:
            # Calcul du score BLEU avec corpus_bleu pour éviter l'erreur liée à fractions
            bleu = corpus_bleu([reference_tokens], [summary_tokens], smoothing_function=smoothing)
        except Exception as e:
            print(f"Erreur lors du calcul du score BLEU : {e}")
            bleu = 0.0

        # Calcul des scores de lisibilité
        fre_score = textstat.flesch_reading_ease(summary)
        dale_chall_score = textstat.dale_chall_readability_score(summary)

        return {
            'rouge1': rouge1,
            'rouge2': rouge2,
            'rougeL': rougeL,
            'bleu': bleu,
            'flesch_reading_ease': fre_score,
            'dale_chall_readability': dale_chall_score
        }

    # Appliquer le calcul des métriques à chaque ligne du DataFrame
    metrics_df = df.apply(compute_metrics, axis=1, result_type='expand')
    return pd.concat([df.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1)

def initialize_metrics(df):
    """
    Initialise les métriques pour les DataFrames fournis.

    Args:
        df (pd.DataFrame): Premier DataFrame à traiter.

    Returns:
        tuple: Deux DataFrames enrichis des métriques calculées.
    """
    # Initialisation du scorer ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Nettoyer et filtrer les DataFrames
    df_cleaned = clean_and_filter_dataframe(df)

    # Calcul des métriques pour les DataFrames filtrés
    df_with_metrics = calculate_metrics(df_cleaned, scorer)

    return df_with_metrics

def calculate_bert_scores(df, summary_col='summary', reference_col='ref_summary', model_type='bert-base-uncased'):
    """
    Calcule le score BERT pour chaque paire de résumé et référence dans le DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les résumés et les résumés de référence.
        summary_col (str): Le nom de la colonne contenant les résumés.
        reference_col (str): Le nom de la colonne contenant les résumés de référence.
        model_type (str): Le type de modèle BERT à utiliser pour le calcul des scores.

    Returns:
        pd.DataFrame: Le DataFrame enrichi avec une colonne 'Bert_Score' contenant les scores calculés.
    """
    # Initialiser le scorer une seule fois pour éviter des appels multiples coûteux
    scorer = BERTScorer(model_type=model_type)

    # Fonction interne pour calculer le BERT score entre un résumé et une référence
    def bert_score(summary, reference):
        try:
            _, _, F1 = scorer.score([summary], [reference])
            return F1.item()  # Convertir le résultat en float
        except Exception as e:
            print(f"Erreur lors du calcul du BERT score : {e}")
            return None

    # Appliquer la fonction bert_score à chaque ligne du DataFrame
    df['bert_score'] = df.apply(lambda x: bert_score(x[summary_col], x[reference_col]), axis=1)

    return df

def postprocess_dataframe(df):
    """
    Effectue le prétraitement du DataFrame en renommant les colonnes, supprimant les colonnes inutiles,
    extrayant et modifiant des valeurs spécifiques de colonnes, et ajustant le format du DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame à prétraiter.

    Returns:
        pd.DataFrame: Le DataFrame modifié et nettoyé.
    """
    # Renommer les colonnes selon les nouveaux noms
    df = df.rename(columns={
        'Bert_Score': 'bert_score',
        'cosine_sim_ref': 'l6_cosine',
        'Accuracy_ref_llm': 'accuracy_ref',
        'Conciseness_ref_llm': 'conciseness_ref',
        'Structure_ref_llm': 'structure_ref',
        'refBased_llm_mean_score': 'ref_based_mean',
        'Accuracy_llm': 'accuracy_free',
        'Clarity_llm': 'clarity_free',
        'Coverage_llm': 'coverage_free',
        'Overall_llm': 'quality_free',
        'flesch_reading_ease': 'flesch_reading',
        'dale_chall_readability': 'dale_chall',
        'refFree_llm_mean_score': 'ref_free_mean'
    })

    # Supprimer la colonne 'sampling' si elle existe
    df = df.drop(['sampling'], axis=1, errors='ignore')

    # Extraire 'random' et 'pre-classified' de la colonne 'type'
    df['sampling'] = df['type'].apply(
        lambda x: 'random' if x.startswith('R_') else 'pre-classified' if x.startswith('PC_') else None
    )

    # Retirer les préfixes 'R_' et 'PC_' de la colonne 'type'
    df['type'] = df['type'].str.replace(r'R_|PC_', '', regex=True)

    # Renommer la colonne 'type' en 'text_type'
    df = df.rename(columns={'type': 'text_type'})

    return df

def plot_interaction_charts(df):
    """
    Affiche trois graphiques d'interaction basés sur les colonnes du DataFrame.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes nécessaires pour les graphiques.
    """
    # Vérifier l'existence des colonnes requises
    required_columns = ['model_name', 'prompt_type', 'bert_score', 'sampling', 'text_type']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"La colonne {col} est manquante dans le DataFrame.")
            return
    
    # Vérifier que les colonnes ne contiennent pas de valeurs manquantes
    for col in required_columns:
        if df[col].isnull().any():
            st.warning(f"Attention : La colonne {col} contient des valeurs manquantes. Elles seront ignorées.")
            df = df.dropna(subset=[col])

    # Créer une figure pour un seul graphique
    fig, ax = plt.subplots(figsize=(10, 6))

    # Tracer les lignes pour relier les points du même type de prompt
    sns.lineplot(data=df, x='model_name', y='bert_score', hue='prompt_type', style='prompt_type', markers=True, ax=ax, legend=True)

    # Ajouter des titres et étiquettes
    ax.set_title("BERT Score par Modèle et Prompt")
    ax.set_xlabel('Modèle de Langage')
    ax.set_ylabel('Mean BERT Score')

    # Afficher la légende
    plt.legend(title='Type de Prompt')

    # Afficher le graphique
    plt.tight_layout()
    st.pyplot(fig)