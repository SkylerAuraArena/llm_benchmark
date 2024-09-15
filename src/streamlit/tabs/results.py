import streamlit as st
import os

title = "Présentation des résultats"
sidebar_name = "Présentation des résultats"

def run():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_path0 = os.path.join(dir_path, "../../assets/images/results.png")
    st.image(img_path0)

    st.title(title)

    st.markdown("---")

    img_path1 = os.path.join(dir_path, "../../assets/slides/Slides Soutenance DataScientest_page-0008.jpg")
    st.image(img_path1)

    img_path2 = os.path.join(dir_path, "../../assets/slides/Slides Soutenance DataScientest_page-0009.jpg")
    st.image(img_path2)