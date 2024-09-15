import streamlit as st
import io
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def calculate_word_statistics(df):
    """
    Affiche des statistiques de base.
    """
    buffer = io.StringIO()  # Crée un buffer StringIO
    df.info(buf=buffer)  # Capture la sortie de df.info() dans le buffer
    s = buffer.getvalue()  # Récupère le contenu du buffer sous forme de chaîne de caractères
    st.text(s) # Affiche le contenu avec Streamlit

    df['word_count'] = df['post'].str.split().str.len().astype('float64')

    st.write("Le jeu de données total contient", df.shape[0], "posts dont", df['post'].duplicated().sum(), " sont des doublons parfaits. Le nombre total de signes est de", df['length'].sum(), "et le nombre total de mots (token primitifs) est de", df['word_count'].sum(), ".")
             
def analyze_text_length(df):
    """
    Analyse la longueur des textes dans le DataFrame.
    """
    df['length'] = df['post'].apply(lambda x: len(str(x)))
    st.write("La longueur des posts varie entre", df['length'].min(), "et", df['length'].max(), "caractères. La longueur moyenne des **posts** est de", df['length'].mean().round(2), "signes et la longueur moyenne des **mots** est de", (df['length'].sum() / df['word_count'].sum()).round(2), "signes par mot, ponctuation incluse.")

    # Affichage des catégories de longueur de texte
    st.write("Nombre de signes pour les posts de mois de 200 signes :", df['length'].loc[df['length'] < 200].count())
    st.write("Nombre de signes pour les posts de moins de 10 000 signes :", df['length'].loc[df['length'] < 10000].count())
    st.write("Nombre de signes pour les posts 10 000 signes ou plus :", df['length'].loc[df['length'] >= 10000].count())

def plot_length_distributions(df):
    """
    Trace les distributions de longueur des posts (histogrammes et boxplots).
    """
    sns.set_theme(font_scale=0.9)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # Distribution globale
    sns.histplot(data=df[['length']], bins=50, facecolor='#A9A5A4', ax=axes[0, 0])
    axes[0, 0].set_xlim(0, 60000)
    axes[0, 0].set_ylim(0, 70000)
    axes[0, 0].set_title('Intégralité des données')

    # Distribution pour < 200 signes
    sns.histplot(data=df['length'].loc[df['length'] < 200], bins=50, facecolor='#7DB87C', ax=axes[0, 1])
    axes[0, 1].set_xlim(0, 200)
    axes[0, 1].set_ylim(0, 6000)
    axes[0, 1].set_title('Moins de 200 signes')

    # Distribution pour < 10 000 signes
    sns.histplot(data=df['length'].loc[df['length'] < 10000], bins=50, facecolor='#A2A7F0', ax=axes[1, 0])
    axes[1, 0].set_xlim(0, 10000)
    axes[1, 0].set_ylim(0, 50000)
    axes[1, 0].set_title('Moins de 10 000 signes')

    # Distribution pour >= 10 000 signes
    sns.histplot(data=df['length'].loc[df['length'] >= 10000], bins=50, facecolor='#F7884D', ax=axes[1, 1])
    axes[1, 1].set_xlim(0, 60000)
    axes[1, 1].set_ylim(0, 80)
    axes[1, 1].set_title('Plus de 10 000 signes')

    plt.tight_layout(pad=1.0)
    st.pyplot(fig)

    # Ceci n'est pas inclus car le chargement est trop long.
    # Boxplot de la longueur des posts
    # fig = go.Figure(data=go.Box(y=df['length'], boxpoints='all', jitter=0.3, pointpos=-1.8))
    # fig.update_layout(
    #     title='Répartition des longueurs de post (Boxplot)',
    #     yaxis_title='Longueur du post'
    # )
    # st.plotly_chart(fig)

def plot_violin_length_distribution(df):
    # Calcul des quartiles Q1 et Q3
    Q1 = df['length'].quantile(0.25)
    Q3 = df['length'].quantile(0.75)

    # Calcul de l'écart interquartile (IQR) et de l'limite haute (Upper Fence UF)
    IQR = Q3 - Q1
    UF = Q3 + 1.5 * IQR

    # Affichage du limite haute
    st.write(f"Limite haute (Upper Fence UF) : {UF}")

    # Création du Violin plot pour la répartition des longueurs de post
    traces = []
    for misinfo, group in df.groupby('misinformation'):
        trace = go.Violin(
            y=group['length'],
            name=f"Misinformation: {misinfo}",
            box_visible=True,
            line_color='black',
            meanline_visible=True,
            fillcolor='lightblue' if misinfo == 0 else 'lightcoral',  # Différencie les couleurs pour 0 et 1
            opacity=0.6,
            hoverinfo='y'
        )
        traces.append(trace)

    # Création de la figure Plotly
    fig = go.Figure(data=traces)
    fig.update_layout(
        title='Répartition des longueurs de post (Violin Plot)',
        yaxis_title='Longueur du post'
    )

    # Défini la plage de l'axe y en fonction de la limite haute
    fig.update_yaxes(range=[0, UF])

    st.plotly_chart(fig)

def plot_target_data(df):
    # Nombre de tweets vides (0 mots)
    empty_tweets_count = df[df['words'] == 0].value_counts().sum()
    # Nombre de tweets avec un seul mot
    single_word_tweets_count = df[df['words'] <= 1].value_counts().sum()

    st.write("Le dataset contient ", empty_tweets_count, " tweets 'vides' et ", single_word_tweets_count, " tweets avec un seul mot.")

def plot_fake_news_distribution(df):
    """
    Trace la répartition de la désinformation (***fake news***) dans le DataFrame puis en fonction de la longueur des posts.
    """
    # Répartition de la désinformation
    plt.figure(figsize=(3, 2))
    sns.catplot(data=df, kind='count', x='misinformation', hue='misinformation', palette={0: 'green', 1: 'red'})
    plt.title('Répartition de la désinformation (0 = information, 1 = désinformation)')
    st.pyplot(plt.gcf())

    # Répartition de la désinformation en fonction de la longueur des posts
    fig = go.Figure()
    fig = px.histogram(df, x='original_id', color="misinformation", color_discrete_map={0: 'green', 1: 'red'})
    st.plotly_chart(fig)