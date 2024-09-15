import streamlit as st
import os

title = "Explication du pré-traitement des données"
sidebar_name = "Explication du pré-traitement des données"

def run():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_path0 = os.path.join(dir_path, "../../assets/images/preprocessing.png")
    st.image(img_path0)

    st.title(title)

    st.markdown("---")

    img_path1 = os.path.join(dir_path, "../../assets/slides/Slides Soutenance DataScientest_page-0004.png")
    st.image(img_path1)

    img_path2 = os.path.join(dir_path, "../../assets/slides/Slides Soutenance DataScientest_page-0005.png")
    st.image(img_path2)