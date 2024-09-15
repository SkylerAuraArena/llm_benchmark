import streamlit as st
import os

title = "Explication de la démarche"
sidebar_name = "Explication de la démarche"

def run():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_path0 = os.path.join(dir_path, "../../assets/images/explication.png")
    st.image(img_path0)

    st.title(title)

    st.markdown("---")

    img_path1 = os.path.join(dir_path, "../../assets/slides/Slides Soutenance DataScientest_page-0001.jpg")
    st.image(img_path1)

    img_path2 = os.path.join(dir_path, "../../assets/slides/Slides Soutenance DataScientest_page-0002.jpg")
    st.image(img_path2)

    img_path3 = os.path.join(dir_path, "../../assets/slides/Slides Soutenance DataScientest_page-0003.jpg")
    st.image(img_path3)