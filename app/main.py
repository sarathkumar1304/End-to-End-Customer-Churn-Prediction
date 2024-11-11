import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from make_prediction import make_prediction
from about import about_me
from project import project_ui
from home import home_page
from EDA import eda
from model import metrics_ui

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Project", "EDA", "Model","About Me"],
        icons=["house", "app-indicator", "bar-chart","person-video" ,"person-video3"],
        menu_icon="cast",
        default_index=1,
    )
if selected == "Project":
    project_ui()
if selected == "Home":
    home_page()

if selected == "EDA":
    eda()

if selected == "Model":
   metrics_ui()

if selected == "About Me":
    about_me()