# Standard library imports
import os
import re
from collections import Counter
from difflib import SequenceMatcher

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from umap.umap_ import UMAP
from wordcloud import WordCloud

# NLTK imports
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

def initialize_app(load_from_remote=False, remote_repo="", local_file=""):
    """
    Initialise l'application en chargeant les données soit depuis un dépôt distant, soit depuis un fichier local.

    Args:
        load_from_remote (bool): Si True, charge les données depuis le dépôt distant; sinon, charge depuis un fichier local.
        remote_repo (str): Le nom du dépôt distant à utiliser si load_from_remote est True.
        local_file (str): Le chemin du fichier local à utiliser si load_from_remote est False.

    Returns:
        None
    """
    try:
        # Choix de la source des données en fonction du paramètre
        if load_from_remote:
            try:
                full_df = initialize_session_state_from_remote_repo(remote_repo)
                print("✅ **Chargement des données depuis le dépôt distant réussi.**")
            except Exception as e:
                st.markdown(f"❌ **Erreur lors du chargement depuis le dépôt distant :** `{e}`")
                return
        else:
            try:
                full_df = load_local_dataset(local_file)
                print("✅ **Chargement des données depuis le fichier local réussi.**")
            except Exception as e:
                st.markdown(f"❌ **Erreur lors du chargement depuis le fichier local :** `{e}`")
                return

        # Prétraitement des données
        try:
            ongoing_df = rename_and_drop_columns(full_df, columns_to_rename={'Unnamed: 0': 'original_id', 'text': 'post', 'label': 'misinformation'}, columns_to_drop=[full_df.columns[0]])
            df = add_information_columns(ongoing_df)
            print("✅ **Prétraitement des données réussi.**")
        except Exception as e:
            st.markdown(f"❌ **Erreur lors du prétraitement des données :** `{e}`")
            return
        
        # Stockage du DataFrame initial pré-traité dans le session_state
        try:
            store_dataframe_in_context(df, key="df")
            print("✅ **Les données ont été stockées dans le contexte de session avec succès.**")
        except Exception as e:
            st.markdown(f"❌ **Erreur lors du stockage des données dans le session_state :** `{e}`")

        st.write("✅ **Chargement des données réussi.**")
    except Exception as e:
        st.markdown(f"❌ **Erreur lors de l'initialisation de l'application :** `{e}`")

def save_dataframe(df, save_as_csv=True, save_as_excel=True, csv_filename='df.csv', excel_filename='df.xlsx'):
    """
    Enregistre un DataFrame en format CSV, Excel, ou les deux.

    Args:
        df (pd.DataFrame): Le DataFrame à enregistrer.
        save_as_csv (bool): Enregistrer en CSV si True. Par défaut, True.
        save_as_excel (bool): Enregistrer en Excel si True. Par défaut, True.
        csv_filename (str): Nom du fichier CSV. Par défaut, 'df.csv'.
        excel_filename (str): Nom du fichier Excel. Par défaut, 'df.xlsx'.

    Returns:
        None
    """
    try:
        if save_as_csv:
            df.to_csv(csv_filename, index=False)
            print(f"DataFrame enregistré en CSV sous le nom '{csv_filename}'.")

        if save_as_excel:
            df.to_excel(excel_filename, index=False)
            print(f"DataFrame enregistré en Excel sous le nom '{excel_filename}'.")
            
    except Exception as e:
        print(f"Erreur lors de l'enregistrement du DataFrame : {e}")

def initialize_session_state_from_local_file(data_dict):
    """
    Initialise le session_state en chargeant les DataFrames et en les fusionnant
    dans un DataFrame unique qui est également stocké dans le session_state.
    """

    if data_dict is None:
        st.warning("Il n'y a pas de dictionnaire de données à charger.")

    # Boucle sur le dictionnaire pour charger et stocker les DataFrames
    for key, address in data_dict.items():
        df = load_local_dataset(address)  # Charge le DataFrame depuis l'adresse spécifiée
        store_dataframe_in_context(df, key=key)  # Stocke le DataFrame dans le session_state avec la clé spécifiée

    # Fusion des DataFrames
    df_combined = pd.concat([st.session_state.df_train, st.session_state.df_test], ignore_index=True)
    return df_combined

def initialize_session_state_from_remote_repo(address = None):
    """
    Initialise le session_state en chargeant les DataFrames depuis une adresse web et en les fusionnant
    dans un DataFrame unique qui est également stocké dans le session_state.
    """

    if address is None:
        st.warning("Il n'y a pas de dictionnaire de données à charger.")
    else:
        ds = load_dataset(address)
        df_train = pd.DataFrame(ds["train"])
        df_test = pd.DataFrame(ds["test"])

        # Fusion des DataFrames
        full_set = pd.concat([df_train, df_test], ignore_index=True)
        return full_set

def store_dataframe_in_context(dataframe=None, key=None):
    """
    Stocke un DataFrame dans le st.session_state avec la clé spécifiée.

    Args:
        dataframe (pd.DataFrame): Le DataFrame à stocker.
        key (str): La clé sous laquelle le DataFrame sera stocké dans le session_state.
    """
    # Initialisation de la clé dans st.session_state si elle n'existe pas
    if key not in st.session_state:
        st.session_state[key] = None

    # Stocke le DataFrame dans le session_state si le DataFrame est fourni
    if dataframe is not None:
        st.session_state[key] = dataframe

def load_local_dataset(filename=None):
    """
    Charge un jeu de données à partir du fichier spécifié, soit CSV, soit Excel.

    Args:
        filename (str, optional): Nom du fichier à charger. Doit être un fichier CSV ou Excel.

    Returns:
        pd.DataFrame: Un DataFrame contenant les données du fichier ou None si le fichier n'est pas trouvé ou ne peut pas être lu.
    """
    if filename is None:
        print("Le nom du fichier n'est pas fourni.")
        return None
    
    # Récupérer le chemin du répertoire du script actuel
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, filename)
    
    # Vérifier si le fichier existe
    if not os.path.isfile(file_path):
        print("Chemin du fichier invalide ou fichier introuvable.")
        return None

    # Déterminer l'extension du fichier et charger en conséquence
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)  # Charger le fichier CSV
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file_path)  # Charger le fichier Excel
        else:
            print("Format de fichier non pris en charge. Veuillez fournir un fichier .csv ou .xlsx.")
            return None
        return df
    except Exception as e:
        print(f"Erreur lors du chargement du fichier : {e}")
        return None

def rename_and_drop_columns(df, columns_to_rename=None, columns_to_drop=None):
    """
    Renomme les colonnes du DataFrame et supprime les colonnes spécifiées.

    Args:
        df (pd.DataFrame): Le DataFrame à manipuler.
        columns_to_rename (dict, optional): Dictionnaire contenant les colonnes à renommer sous forme {ancien_nom: nouveau_nom}.
        columns_to_drop (list, optional): Liste des colonnes à supprimer.

    Returns:
        pd.DataFrame: Le DataFrame avec les colonnes renommées et supprimées.
    """
    # Renomme les colonnes si spécifiées
    if columns_to_rename:
        df = df.rename(columns=columns_to_rename)

    # Supprime les colonnes spécifiées
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')  # errors='ignore' pour éviter les erreurs si une colonne n'existe pas

    return df

def add_information_columns(df):
    """
    Ajoute des colonnes supplémentaires au DataFrame pour la longueur des caractères et le nombre de mots.

    Args:
        df (pd.DataFrame): Le DataFrame à manipuler.

    Returns:
        pd.DataFrame: Le DataFrame avec les colonnes supplémentaires ajoutées.
    """

    # Convertir les éléments de la colonne 'post' en chaînes de caractères pour éviter les erreurs avec len()
    df['post'] = df['post'].astype(str)

    # Ajouter une colonne 'length' pour la longueur en caractères du texte
    df['length'] = df['post'].apply(len)
    
    # Ajouter une colonne 'words' pour le nombre de mots
    df['words'] = df['post'].apply(lambda x: len(x.split()))
    
    return df

def get_primitive_elts(df):
    """
    Récupère les mots primitifs du DataFrame en supprimant des caractères spéciaux.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.

    Returns:
        list: Une liste de mots primitifs.
    """
    # Suppression des caractères spéciaux et séparation des mots
    df['primitive_words'] = df['post'].str.replace(r'[^\w\s]', '', regex=True).str.split()
    
    # Concaténation des mots primitifs
    primitive_elts = [word for sublist in df['primitive_words'].tolist() for word in sublist]
    
    # Calcul des statistiques sur les mots primitifs
    word_counts = Counter(primitive_elts)
    all_words = primitive_elts

    # Mot le plus fréquent et le moins fréquent
    keyMax = list(word_counts.keys())[list(word_counts.values()).index(max(list(word_counts.values())))]
    keyMin = list(word_counts.keys())[list(word_counts.values()).index(min(list(word_counts.values())))]

    # Affichage des statistiques avec Streamlit
    st.write("Le nombre total de mots primitifs dans le corpus est de ", len(all_words), " dont ", len(word_counts), " mots primitifs uniques. La moyenne des occurrences par mot primitif unique est de ", round(len(all_words)/len(word_counts), 2), " sachant que l'occurences maximum est de ", max(list(word_counts.values())), " et l'occurences minimum est de ", min(list(word_counts.values())), ".")

    st.write("À noter que le mot le plus fréquent est ", {keyMax}, " et le mot le moins fréquent est ", {keyMin}, ".")

def text_pruning(df):
    """
    Élagage des scories textuelles dans le DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.

    Returns:
        pd.DataFrame: Le DataFrame avec les scories élaguées.
    """
    # Suppression des doublons
    pruned_df = df.drop_duplicates(subset=['post'], keep='first')
    deleted_posts = len(df) - len(pruned_df)

    st.write("La traitement a permis de supprimer les ", deleted_posts, " doublons les plus évidents et il reste ", len(pruned_df), " dans le jeu de données.")

    return pruned_df

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Fonctions de nettoyage des textes
def remove_urls(text):
    if isinstance(text, str):
        url_pattern = re.compile(r'http[s]?://\S+|www\.\S+')
        return url_pattern.sub('', text)

def remove_emails(text):
    if isinstance(text, str):
        email_pattern = re.compile(r'([a-zA-Z0-9_\.-]+)@([a-zA-Z0-9_\.-]+)\.([a-zA-Z]{2,5})')
        return email_pattern.sub('', text)

def remove_gobbledegook(text):
    if isinstance(text, str):
        gobbledegook = re.compile(r'\b(?=[a-zA-Z0-9]*[A-Z])(?=[a-zA-Z0-9]*[a-z])(?=[a-zA-Z0-9]*\d)[a-zA-Z0-9]{8,12}\b')
        return gobbledegook.sub(' ', text)

def remove_at_username(text):
    if isinstance(text, str):
        at_username = re.compile(r'@([a-zA-Z0-9_\.-]+)')
        return at_username.sub(' ', text)

def remove_hashtag(text):
    if isinstance(text, str):
        hashtag = re.compile(r'#([a-zA-Z0-9_\.-]+)')
        return hashtag.sub(' ', text)

def remove_spe_car_exclu_comma_dot(text):
    if isinstance(text, str):
        remove_punctuation = re.compile(r'[^\w\s]')
        return remove_punctuation.sub(' ', text)

def clean_text(df):
    """
    Nettoie le texte dans la colonne 'text' et crée une nouvelle colonne 'text_prepr' avec le texte nettoyé.

    Args:
        df (pd.DataFrame): Le DataFrame contenant la colonne 'text'.

    Returns:
        pd.DataFrame: Le DataFrame avec la colonne 'text_prepr' nettoyée.
    """
    # Création de la colonne 'text_prepr' en modifiant le texte original
    df.loc[:, 'post_prepr'] = df['post'].str.replace('WASHINGTON', ' ')
    df.loc[:, 'post_prepr'] = df['post_prepr'].str.replace(r'Reuters|reuters|REUTERS', ' ', regex=True)
    df.loc[:, 'post_prepr'] = df['post_prepr'].str.replace(r'Ä¶|Äô|Äù|Äú|Å©|äî|Äî', ' ', regex=True)
    df.loc[:, 'post_prepr'] = df['post_prepr'].str.replace(r'\bu\b', ' ', regex=True)
    df.loc[:, 'post_prepr'] = df['post_prepr'].str.replace(r'\bs\b', ' ', regex=True)
    df.loc[:, 'post_prepr'] = df['post_prepr'].str.replace(r'\br\b', ' ', regex=True)

    # Appliquer toutes les fonctions de nettoyage
    df.loc[:, 'post_prepr'] = df['post_prepr'].apply(remove_urls)
    df.loc[:, 'post_prepr'] = df['post_prepr'].apply(remove_emails)
    df.loc[:, 'post_prepr'] = df['post_prepr'].apply(remove_gobbledegook)
    df.loc[:, 'post_prepr'] = df['post_prepr'].apply(remove_at_username)
    df.loc[:, 'post_prepr'] = df['post_prepr'].apply(remove_hashtag)
    df.loc[:, 'post_prepr'] = df['post_prepr'].apply(remove_spe_car_exclu_comma_dot)

    # Nettoyage supplémentaire
    df.loc[:, 'post_prepr'] = df['post_prepr'].str.replace('.', '. ')
    df.loc[:, 'post_prepr'] = df['post_prepr'].str.replace('  ', ' ')
    df.loc[:, 'post_prepr'] = df['post_prepr'].str.replace('   ', ' ')

    st.write("✅ Le jeu de données a été nettoyé avec succès.")

    return df

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def display_wordcloud(wordcloud):
    """
    Affiche un nuage de mots dans un graphique matplotlib.

    Args:
        wordcloud (WordCloud): Objet WordCloud à afficher.
    """
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

def generate_wordcloud_from_dataframe(df, column_name, max_words=20):
    """
    Génère et affiche un nuage de mots à partir d'une colonne spécifique d'un DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données textuelles.
        column_name (str): Le nom de la colonne contenant les posts.
        max_words (int): Le nombre maximum de mots à inclure dans le nuage de mots.
    """

    with st.spinner(f'Traitement en cours... Merci de patienter.'):
        # Initialiser les stopwords pour l'anglais
        stop_words_english = set(nltk.corpus.stopwords.words('english'))
    
    # Utiliser un spinner pour indiquer que le processus est en cours
    with st.spinner(f'Génération du nuage des {max_words} mots en cours...'):
        # Concaténer tous les textes de la colonne spécifiée en une seule chaîne
        text = " ".join(df[column_name].dropna().astype(str))

        # Générer le nuage de mots
        wordcloud = WordCloud(
            background_color="black",
            max_words=max_words,
            stopwords=stop_words_english,
            max_font_size=50
        ).generate(text)

    # Afficher le nuage de mots avec Streamlit
    st.subheader(f"Nuage de {max_words} mots les plus fréquents")
    display_wordcloud(wordcloud)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def compute_similarity(text1, text2):
    """
    Calcule la similarité de deux textes en utilisant la distance de Levenshtein.

    Args:
        text1 (str): Premier texte.
        text2 (str): Deuxième texte.

    Returns:
        float: Ratio de similarité entre les deux textes.
    """
    return SequenceMatcher(None, text1, text2).ratio()

def flag_similar_posts(df, custom_dataset=False, threshold=0.9, max_features=15000):
    """
    Flague les lignes avec une similarité lexicale supérieure au seuil donné.

    Args:
        df (pd.DataFrame): DataFrame contenant les textes à analyser.
        threshold (float): Seuil de similarité pour flaguer les textes.
        max_features (int): Nombre maximum de caractéristiques pour le vecteur TF-IDF.
        use_original_id (bool): Si True, utilise 'original_id' pour 'similar_with', sinon utilise l'index.

    Returns:
        pd.DataFrame: DataFrame avec les colonnes de similarité ajoutées.
    """

    with st.spinner('Traitement de la similarité en cours... Merci de patienter.'):
        # Initialisation des colonnes nécessaires
        df['has_similarity'] = 0
        df['similar_with'] = None
        df['similarity_group'] = None
        df['similarity_score'] = 0.0

        # Calcul du TF-IDF
        vectorizer = TfidfVectorizer(max_features=max_features).fit_transform(df['post_prepr'])
        vectors = vectorizer.toarray()

        # Calcul de la similarité cosinus
        cosine_sim = cosine_similarity(vectors)

        group_counter = 0
        visited = np.zeros(len(df), dtype=bool)

        for i in range(len(df)):
            if visited[i]:
                continue

            # Identifie les textes similaires
            similar_indices = np.where(cosine_sim[i] >= threshold)[0]
            if len(similar_indices) > 1:
                group_counter += 1

                for j in similar_indices:
                    if i != j:
                        df.at[j, 'has_similarity'] = 1
                        # Choix de 'similar_with' selon le paramètre 'use_original_id'
                        df.at[j, 'similar_with'] = df.index[i] if custom_dataset else df.at[i, 'original_id']
                        df.at[j, 'similarity_score'] = cosine_sim[i][j]
                        df.at[j, 'similarity_group'] = group_counter
                        visited[j] = True

                # Pour le premier texte lui-même
                df.at[i, 'has_similarity'] = 1
                df.at[i, 'similarity_score'] = 1.0  # Car il est identique à lui-même
                df.at[i, 'similarity_group'] = group_counter

        return df

def remove_duplicated_from_project_dataset(df):
    """
    Supprime les doublons du jeu de données projeté en identifiant et en éliminant les posts similaires.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
    
    Cette fonction :
    - Trie les posts par longueur.
    - Divise le jeu de données en deux parties pour l'analyse.
    - Flague les posts similaires dans chaque partie.
    - Concatène et retrie les résultats pour une analyse plus précise.
    - Génère le jeu de données final sans doublons identifiés.

    Returns:
        pd.DataFrame: Le DataFrame sans les doublons.
    """

    # Tri des données par longueur de texte
    df = df.sort_values(by='length')

    # Division du DataFrame en deux parties pour traiter les doublons séparément
    df_1_2 = df[:43000].reset_index(drop=True)
    df_2_2 = df[43000:].reset_index(drop=True)

    # Identification des doublons dans chaque partie
    first_part = flag_similar_posts(df_1_2)

    st.markdown("✅ **Traitement de la première moitié du jeu de données terminé.**")

    second_part = flag_similar_posts(df_2_2)

    st.markdown("✅ **Traitement de la seconde moitié du jeu de données terminé.**")

    # Concaténation des deux parties pour une analyse globale
    wholedf = pd.concat([first_part, second_part]).reset_index(drop=True)

    # Division du DataFrame complet en trois tiers pour une vérification plus fine des doublons
    df_1tiers = wholedf[0:25392].reset_index(drop=True)
    df_CENTER = wholedf[25392:50784].reset_index(drop=True)
    df_3tiers = wholedf[50784:].reset_index(drop=True)

    # Réanalyse du centre du DataFrame pour identifier les doublons
    center_df = flag_similar_posts(df_CENTER)

    st.markdown("✅ **Traitement final du jeu de données terminé.**")

    # Concaténation finale des trois parties
    df_final = pd.concat([df_1tiers, center_df, df_3tiers], axis=0).reset_index(drop=True)
    
    # Calcul des statistiques de réduction de doublons
    duplicated = df_final[(df_final['has_similarity'] == 1) & ~(df_final['similar_with'].isna())]
    deleted = len(duplicated)

    # Extraction des indices des lignes marquées comme doublons
    duplicated_indices = df_final[
        (df_final['has_similarity'] == 1) & 
        ~(df_final['similar_with'].isna())
    ].index

    # Suppression des lignes dupliquées de df_final en utilisant les indices
    df_final = df_final.drop(index=duplicated_indices)

    st.markdown("✅ **Traitement des similarités terminé.**")

    # Affichage des résultats de l'analyse de doublons
    st.write("Le nombre de smili-doublons supprimés est de " , deleted, ". Le jeu de données ne comporte à présent plus que ", len(df_final), " lignes.")

    return df_final

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def remove_duplicated_from_custom_dataset(df_custom):
    """
    Supprime les doublons du jeu de données projeté en identifiant et en éliminant les posts similaires.

    Args:
        df_custom (pd.DataFrame): Le DataFrame contenant les données.

    Cette fonction :
    - Trie les posts par longueur.
    - Flague les posts similaires dans chaque partie.
    - Génère le jeu de données final sans doublons identifiés.

    Returns:
        pd.DataFrame: Le DataFrame sans les doublons.
    """

    # Tri des données par longueur de texte
    df_custom = df_custom.sort_values(by='length')

    # Identification des doublons dans chaque partie
    df_custom = flag_similar_posts(df_custom, custom_dataset=True)
    
    # Calcul des statistiques de réduction de doublons
    duplicated = df_custom[(df_custom['has_similarity'] == 1) & ~(df_custom['similar_with'].isna())]
    deleted = len(duplicated)

    # Extraction des indices des lignes marquées comme doublons
    duplicated_indices = df_custom[
        (df_custom['has_similarity'] == 1) & 
        ~(df_custom['similar_with'].isna())
    ].index

    # Suppression des lignes dupliquées de df_custom en utilisant les indices
    df_custom_final = df_custom.drop(index=duplicated_indices)

    st.markdown("✅ **Traitement accompli avec succès.**")

    # Affichage des résultats de l'analyse de doublons
    st.write("Le nombre de smili-doublons supprimés est de " , deleted, ". Le jeu de données ne comporte à présent plus que ", len(df_custom_final), " lignes.")

    return df_custom_final

def selected_size_picking(df, custom_dataset=False, lower_bound = 50, upper_bound = 3000):
    """
    Sélectionne un échantillon de données en fonction des limites de longueur spécifiées.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        lower_bound (int): Limite inférieure de longueur de texte.
        upper_bound (int): Limite supérieure de longueur de texte.

    Returns:
        pd.DataFrame: Le DataFrame avec les lignes sélectionnées.
    """
    # Sélection des lignes avec une longueur de texte dans la plage spécifiée
    filtered_df = df[(df['length'] >= lower_bound) & (df['length'] <= upper_bound)]

    taille_originale = len(st.session_state.df_custom) if custom_dataset else len(st.session_state.df)
    reduction = np.round((taille_originale - len(filtered_df)) / taille_originale * 100, 2)

    st.write(f"En sélectionnant uniquement les textes dont la longueur est comprise entre ", lower_bound, " et ", upper_bound, " caractères, le jeu de données a été réduit à ", len(filtered_df), " lignes, soit ", reduction, "% du jeu de données d'origine qui était de ", taille_originale, " lignes.")

    # Réinitialisation de l'index du DataFrame
    filtered_df.reset_index(drop=True, inplace=True)

    # Utilisation de .loc pour créer une nouvelle colonne 'id_50_3000' avec les valeurs de l'index
    filtered_df.loc[:, 'id_50_3000'] = filtered_df.index

    # Réorganisation des colonnes pour mettre 'id_50_3000' en premier si nécessaire
    filtered_df = filtered_df[['id_50_3000', 'original_id', 'post', 'misinformation', 'length', 'words', 'primitive_words', 'post_prepr', 'has_similarity', 'similar_with', 'similarity_group', 'similarity_score']]
    
    return filtered_df

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def interactive_cluster_selection(cluster_options = [100]):
    """
    Permet à l'utilisateur de sélectionner de manière interactive les valeurs de n_clusters à tester.

    Args:
        cluster_options (list): Liste des options de nombres de clusters à proposer.

    Returns:
        list: Liste des nombres de clusters sélectionnés par l'utilisateur.
    """

    # Utiliser un multiselect pour permettre à l'utilisateur de choisir les valeurs
    selected_clusters = st.multiselect(
        'Sélectionnez les nombres de clusters à tester :',
        options=cluster_options,
        default=[]  # Valeurs par défaut sélectionnées
    )

    # Retourner les valeurs sélectionnées pour les utiliser dans d'autres fonctions
    return selected_clusters

def make_SBERT_vectorization(df, column_name='post_prepr', model_name='all-MiniLM-L6-v2'):
    """
    Génère les vecteurs SBERT pour les textes d'un DataFrame avec une barre de progression Streamlit.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les textes à vectoriser.
        column_name (str): Le nom de la colonne contenant les textes.
        model_name (str): Le nom du modèle SBERT à utiliser.

    Returns:
        np.array: Un tableau de vecteurs SBERT pour les textes.
    """
    # Récupération des longueurs des textes
    # df['prepr_sign_count'] = df[column_name].apply(lambda x: len(str(x)))

    # Chargement du modèle SBERT
    model = SentenceTransformer(model_name)

    # Initialiser la barre de progression Streamlit
    progress_bar = st.progress(0)
    total = len(df)  # Nombre total de textes

    # Diviser les textes en lots pour afficher la progression
    batch_size = 10  # Taille du lot à ajuster selon vos besoins
    vectors = []

    # Générer les vecteurs SBERT pour les textes par lots
    for i in range(0, total, batch_size):
        batch_texts = df[column_name].values[i:i + batch_size]
        batch_vectors = model.encode(batch_texts)
        vectors.extend(batch_vectors)
        
        # Mettre à jour la barre de progression
        progress_bar.progress(min((i + batch_size) / total, 1.0))

    store_dataframe_in_context(np.array(vectors), key="doc_emb")

    return np.array(vectors)

def make_Kmeans_clustering(df, n_clusters_list=[3], init='k-means++', n_init=10, max_iter=300, random_state=1234):
    """
    Effectue la vectorisation SBERT des textes puis le clustering KMeans pour chaque valeur de n_clusters spécifiée,
    en ajoutant les résultats en tant que colonnes du DataFrame, avec une barre de progression Streamlit.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les vecteurs SBERT.
        n_clusters_list (list): Liste des nombres de clusters à tester.
        init (str): Méthode pour initialiser les centroids.
        n_init (int): Nombre d'initialisations différentes à essayer.
        max_iter (int): Nombre maximum d'itérations pour chaque initialisation.
        random_state (int): La graine aléatoire pour initialiser le générateur.

    Returns:
        pd.DataFrame: Le DataFrame avec les colonnes des labels de clusters ajoutées.
    """

    # Vectorisation SBERT des textes
    doc_emb = make_SBERT_vectorization(df)

    st.markdown("✅ **Vectorisation des données réalisée avec succès.**")

    # Initialiser la barre de progression Streamlit
    progress_bar = st.progress(0)
    total_clusters = len(n_clusters_list)  # Nombre total de configurations de clustering

    # Boucle sur chaque valeur de n_clusters spécifiée
    for i, n_clusters in enumerate(n_clusters_list):
        # Initialisation du modèle KMeans
        kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, random_state=random_state)

        # Ajustement du modèle aux vecteurs SBERT
        kmeans.fit(doc_emb)

        # Ajouter une nouvelle colonne avec les labels KMeans
        column_name = f'km_labels{n_clusters}'
        df[column_name] = kmeans.labels_

        # Mettre à jour la barre de progression
        progress_bar.progress((i + 1) / total_clusters)

    return df

def plot_umap_clusters(df, embedding_column_prefix='km_labels', n_neighbors=15, min_dist=0.1, metric='euclidean'):
    """
    Affiche une UMAP des différentes clusterisations présentes dans le DataFrame avec Streamlit.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les vecteurs SBERT et les labels de clusterisation.
        embedding_column_prefix (str): Préfixe commun des colonnes contenant les labels de clusters.
        n_neighbors (int): Nombre de voisins à considérer pour UMAP.
        min_dist (float): Distance minimale pour UMAP.
        metric (str): Métrique de distance à utiliser pour UMAP.
    """

    # Sélectionner les colonnes de labels de clustering
    cluster_columns = [col for col in df.columns if col.startswith(embedding_column_prefix)]
    
    if not cluster_columns:
        st.warning("Aucune colonne de clusterisation trouvée.")
        return

    # Vectorisation SBERT des textes
    doc_emb = make_SBERT_vectorization(df)

    # Réduction de la dimensionnalité avec UMAP
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    embedding = reducer.fit_transform(doc_emb)

    # Affichage des clusters
    st.title('Visualisation UMAP des Clusterisations')

    # Créer une figure pour chaque clusterisation
    for cluster_column in cluster_columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(
            embedding[:, 0], embedding[:, 1], 
            c=df[cluster_column], cmap='Spectral', s=10, alpha=0.7
        )
        plt.title(f'Clusterisation avec {cluster_column}')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.colorbar(label='Cluster Label')
        st.pyplot(plt)
        plt.close()

def evaluate_clusters(df, doc_emb, embedding_column_prefix='km_labels'):
    """
    Calcule et affiche le Davies-Bouldin Index (DBI) et le Calinski-Harabasz Index 
    pour chaque clusterisation présente dans le DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les labels de clusters.
        doc_emb (np.array): Les vecteurs SBERT des textes.
        embedding_column_prefix (str): Préfixe commun des colonnes contenant les labels de clusters.
    """

    # Sélectionner les colonnes de labels de clustering
    cluster_columns = [col for col in df.columns if col.startswith(embedding_column_prefix)]
    
    if not cluster_columns:
        st.warning("Aucune colonne de clusterisation trouvée.")
        return

    # Afficher les résultats dans Streamlit
    st.title('Évaluation des Clusterisations')

    # Calculer et afficher les indices pour chaque clusterisation
    results = []
    for cluster_column in cluster_columns:
        labels = df[cluster_column]
        dbi = davies_bouldin_score(doc_emb, labels)
        ch_index = calinski_harabasz_score(doc_emb, labels)
        
        # Stocker les résultats dans une liste pour un affichage plus structuré si nécessaire
        results.append({
            'Cluster': cluster_column,
            'Davies-Bouldin Index': dbi,
            'Calinski-Harabasz Index': ch_index
        })
    
    # Optionnel : Afficher les résultats sous forme de tableau dans Streamlit
    st.dataframe(pd.DataFrame(results))

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def classify_texts_and_split(df, column_name='length'):
    """
    Extrait le 'type' des textes ('short' ou 'medium') en fonction du nombre de caractères 
    et crée deux DataFrames distincts pour chaque type.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les textes et leur comptage de caractères.
        column_name (str): Le nom de la colonne contenant le nombre de caractères.

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame: Le DataFrame original modifié avec la colonne 'type',
        et deux DataFrames distincts pour les textes 'short' et 'medium'.
    """
    # Ajout de la colonne 'type' pour classer les textes
    df['type'] = df[column_name].apply(
        lambda x: 'short' if 50 <= x <= 280 else 'medium' if x > 280 else None
    )

    # Création des DataFrames pour chaque type
    df_short = df[df['type'] == 'short']
    df_medium = df[df['type'] == 'medium']

    return df, df_short, df_medium

def count_tokens(text):
    """
    Compte le nombre de tokens dans un texte en utilisant le tokenizer de NLTK.

    Args:
        text (str): Le texte à tokeniser.

    Returns:
        int: Le nombre de tokens dans le texte.
    """
    return len(nltk.word_tokenize(text))

def create_randomized_clusters(df, max_clusters=100, max_tokens_per_cluster=1900):
    """
    Crée des clusters de textes aléatoires en respectant un nombre maximum de clusters et une limite de tokens par cluster.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les textes à clusteriser.
        max_clusters (int): Nombre maximum de clusters à créer.
        max_tokens_per_cluster (int): Limite de tokens par cluster.

    Returns:
        pd.DataFrame: Un DataFrame contenant les clusters créés.
    """
    clusters = []
    cluster_id = 1
    cluster_text, clustered_ids = [], []
    cluster_tokens = 0

    # Mélanger le DataFrame de manière aléatoire
    df = df.sample(frac=1).reset_index(drop=True)

    # Parcourir chaque ligne du DataFrame
    for _, row in df.iterrows():
        text = f"- {row['post']}\n"
        text_tokens = count_tokens(text)

        # Si la limite de tokens est dépassée, sauvegarder le cluster et réinitialiser
        if cluster_tokens + text_tokens > max_tokens_per_cluster:
            clusters.append({
                'cluster_id': cluster_id,
                'clustered_text': ''.join(cluster_text),
                'id_50_3000': ', '.join(clustered_ids),
                'token_number': cluster_tokens
            })
            cluster_id += 1
            cluster_text, clustered_ids = [], []
            cluster_tokens = 0
            if cluster_id > max_clusters:
                break

        # Ajouter le texte et ses tokens au cluster actuel
        cluster_text.append(text)
        cluster_tokens += text_tokens
        clustered_ids.append(str(row['id_50_3000']))

    # Ajouter le dernier cluster restant
    if cluster_text:
        clusters.append({
            'cluster_id': cluster_id,
            'clustered_text': ''.join(cluster_text),
            'id_50_3000': ', '.join(clustered_ids),
            'token_number': cluster_tokens
        })

    return pd.DataFrame(clusters)

def add_type_and_modify_cluster_id(df, cluster_type):
    """
    Ajoute une colonne 'type' et modifie le 'cluster_id' en ajoutant un préfixe spécifique.

    Args:
        df (pd.DataFrame): Le DataFrame à modifier.
        cluster_type (str): Le type à ajouter ('short' ou 'medium').

    Returns:
        pd.DataFrame: Le DataFrame modifié avec les colonnes 'type' et 'cluster_id' mises à jour.
    """
    df['type'] = f'R_{cluster_type}'
    df['cluster_id'] = df['type'] + '_' + df['cluster_id'].astype(str)
    return df

def create_topic_clusters(df, max_tokens_per_cluster=1900):
    """
    Crée des clusters de textes par topic en respectant une limite de tokens.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les textes et topics.
        max_tokens_per_cluster (int): Limite de tokens par cluster.

    Returns:
        pd.DataFrame: DataFrame des clusters créés par topic.
    """
    clusters = []
    cluster_id = 1

    for topic in df['topic'].unique():
        topic_df = df[df['topic'] == topic].sample(frac=1).reset_index(drop=True)
        cluster_text = []
        cluster_tokens = 0
        clustered_ids = []

        for _, row in topic_df.iterrows():
            text = f"- {row['post']}\n"
            text_tokens = count_tokens(text)

            if cluster_tokens + text_tokens > max_tokens_per_cluster:
                break

            cluster_text.append(text)
            cluster_tokens += text_tokens
            clustered_ids.append(str(row['id_50_3000']))

        if cluster_text:
            clusters.append({
                'cluster_id': cluster_id,
                'clustered_text': ''.join(cluster_text),
                'id_50_3000': ', '.join(clustered_ids),
                'token_number': cluster_tokens,
                'topic': topic
            })
            cluster_id += 1

    return pd.DataFrame(clusters)

def filter_and_process_clusters(clusters, posts, min_tokens=1000, cluster_type='short', max_posts=100):
    """
    Filtre et traite les clusters et les posts, en éliminant les mauvais topics et IDs utilisés.

    Args:
        clusters (pd.DataFrame): DataFrame des clusters.
        posts (pd.DataFrame): DataFrame des posts à filtrer.
        min_tokens (int): Nombre minimum de tokens pour conserver un cluster.
        cluster_type (str): Type de cluster ('short' ou 'medium').
        max_posts (int): Nombre maximum de posts à inclure.

    Returns:
        pd.DataFrame: Clusters finaux ajustés et combinés.
    """
    # Filtrer les clusters avec suffisamment de tokens
    filtered_clusters = clusters[clusters['token_number'] >= min_tokens]

    # Identifier les mauvais topics et filtrer les posts correspondants
    bad_topics = clusters[clusters['token_number'] < min_tokens]['topic']
    ids_to_remove = filtered_clusters['id_50_3000'].str.split(', ').explode().astype(int).tolist()
    filtered_posts = posts[~posts['topic'].isin(bad_topics)]
    filtered_posts = filtered_posts[~filtered_posts['id_50_3000'].isin(ids_to_remove)]

    # Créer de nouveaux clusters avec les posts filtrés
    fill_clusters = create_topic_clusters(filtered_posts)
    tokens_deleted = len(clusters) - len(filtered_clusters)
    fill_clusters = fill_clusters.head(tokens_deleted + 1)

    # Combiner les clusters filtrés et nouveaux pour atteindre le nombre souhaité
    final_clusters = pd.concat([filtered_clusters, fill_clusters], ignore_index=True).head(max_posts)
    final_clusters['type'] = cluster_type
    final_clusters['cluster_id'] = [f"PC_{cluster_type}_{i}" for i in range(1, len(final_clusters) + 1)]

    return final_clusters