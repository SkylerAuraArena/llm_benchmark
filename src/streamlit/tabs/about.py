import streamlit as st
import os

title = "À propos"
sidebar_name = "À propos"

def run():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(dir_path, "../../assets/images/team.png")
    st.image(img_path)

    st.title(title)

    st.markdown("---")

    st.write("### Rapport et synthèse")

    st.write("Le rapport intégral de notre travail est disponible à l'adresse suivante : https://drive.proton.me/urls/PTD9EN8B2W#BGitw516cRP9")
    st.write("Une synthèse du rapport est disponible ici : https://drive.proton.me/urls/REQ1HD46V0#Gc5viyqFuMEM")

    st.write("### Équipe projet")

    # HTML pour intégrer le logo et le lien
    html_template_github = """<div class="divrefs"><span class="spanrefs">{}</span> : <a href="{}" target="_blank">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#0a66c2" width="24" height="24"><path d="M20.5 2h-17A1.5 1.5 0 002 3.5v17A1.5 1.5 0 003.5 22h17a1.5 1.5 0 001.5-1.5v-17A1.5 1.5 0 0020.5 2zM8 19H5v-9h3zM6.5 8.25A1.75 1.75 0 118.3 6.5a1.78 1.78 0 01-1.8 1.75zM19 19h-3v-4.74c0-1.42-.6-1.93-1.38-1.93A1.74 1.74 0 0013 14.19a.66.66 0 000 .14V19h-3v-9h2.9v1.3a3.11 3.11 0 012.7-1.4c1.55 0 3.36.86 3.36 3.66z"></path></svg></a><a href="{}" target="_blank"><svg height="24" viewBox="0 0 16 16" width="24" fill="black"><path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"></path></svg></a>
            </div>
    """
    html_template_no_github = """<div class="divrefs"><span class="spanrefs">{}</span> : <a href="{}" target="_blank">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#0a66c2" width="24" height="24"><path d="M20.5 2h-17A1.5 1.5 0 002 3.5v17A1.5 1.5 0 003.5 22h17a1.5 1.5 0 001.5-1.5v-17A1.5 1.5 0 0020.5 2zM8 19H5v-9h3zM6.5 8.25A1.75 1.75 0 118.3 6.5a1.78 1.78 0 01-1.8 1.75zM19 19h-3v-4.74c0-1.42-.6-1.93-1.38-1.93A1.74 1.74 0 0013 14.19a.66.66 0 000 .14V19h-3v-9h2.9v1.3a3.11 3.11 0 012.7-1.4c1.55 0 3.36.86 3.36 3.66z"></path></svg></a></div>
    """

    # Écrire les liens avec le logo
    st.markdown(html_template_github.format("Étienne Breton", "https://www.linkedin.com/in/etienne-breton-audit-data-blockchain","https://github.com/SkylerAuraArena"), unsafe_allow_html=True)
    st.markdown(html_template_no_github.format("Willy Fux", "https://www.linkedin.com/in/willy-fux-12807217/"), unsafe_allow_html=True)
    st.markdown(html_template_no_github.format("Jean-Victor Steinlein", "https://www.linkedin.com/in/jeanvictor-steinlein/"), unsafe_allow_html=True)
    st.markdown(html_template_no_github.format("Guillaume Zanini", "https://www.linkedin.com/in/guillaume-zanini-8769bb76/"), unsafe_allow_html=True)

    st.write("")
    st.write("### Sources")

    # Écrire les liens avec le logo
    st.write("Les différentes illusatrations ont été générées par le logiciel ***DALL-E*** développé par ***OpenAI*** (https://openai.com/blog/dall-e/).")
    st.write("Le jeu de données a été mis à disposition sur le site ***Huggingface*** à l'adresse suivante : https://huggingface.co/datasets/roupenminassian/twitter-misinformation.")
    st.write("Ce projet a été réalisé dans le cadre du cursus ***Data Scientist*** de ***Datascientest***. Plus d'information à l'adresse suivante : https://datascientest.com/formation-data-scientist.")