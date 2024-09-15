import streamlit as st
import os

title = "Conclusion"
sidebar_name = "Consclusion"

def run():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(dir_path, "../../assets/images/conclusion.png")
    st.image(img_path)

    st.title(title)

    st.markdown("---")

    img_path0 = os.path.join(dir_path, "../../assets/slides/Slides Soutenance DataScientest_page-0007.png")
    st.image(img_path0)