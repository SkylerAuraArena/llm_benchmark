from collections import OrderedDict
import streamlit as st
import os

from tabs import introduction, explication_projet, explication_preprocessing, live_preprocessing, explication_llm, summurization, results, conclusion, about

svg = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 300" class="logo">
    <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-family="Arial, sans-serif" font-size="120" fill="#fff">
        LLM
    </text>
    <text x="50%" y="75%" dominant-baseline="middle" text-anchor="middle" font-family="Arial, sans-serif" font-size="40" fill="#fff">
        BENCHMARK
    </text>
</svg>
"""

st.set_page_config(
    page_title= 'Language Models Summurization Benchmark',
    page_icon="https://img.freepik.com/premium-vector/artificial-intelligence-ai-icon-flat-vector-illustration_423491-69.jpg",
)

# Chemin absolu vers le dossier contenant app.py
dir_path = os.path.dirname(os.path.realpath(__file__))
css_file_path = os.path.join(dir_path, "style.css")

with open(css_file_path, "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

# TODO: add new and/or renamed tab in this ordered dict by
# passing the name in the sidebar as key and the imported tab
# as value as follow :
TABS = OrderedDict(
    [
        (introduction.sidebar_name, introduction),
        (explication_projet.sidebar_name, explication_projet),
        (explication_preprocessing.sidebar_name, explication_preprocessing),
        (explication_llm.sidebar_name, explication_llm),
        (live_preprocessing.sidebar_name, live_preprocessing),
        (summurization.sidebar_name, summurization),
        (results.sidebar_name, results),
        (conclusion.sidebar_name, conclusion),
        (about.sidebar_name, about),
    ]
)

def run():

    st.sidebar.markdown(svg, unsafe_allow_html=True)
    
    tab_name = st.sidebar.radio("", list(TABS.keys()), 0)
    
    tab = TABS[tab_name]

    tab.run()


if __name__ == "__main__":
    run()
