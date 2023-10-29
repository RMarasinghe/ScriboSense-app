import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from summarize_page import show_summarize_page
# from login_page import show_login_page
from transformers import BertTokenizer
import joblib
import torch
import pyrebase
from dashboard import show_dashboard
# from pages.Login import logout

# ------------Load the models-----------------------------------------
# model_data = joblib.load("wording_model_2/wording_regressor_model.pkl")

# regressor = model_data.get("regressionModel")

#     # Load the GPT model and tokenizer
# gpt2_model, tokenizer = load_gpt_model()

# --------------------------------------------------------------------

#--------------------

st.markdown("""
<style>
.css-1rs6os.edgvbvh3
{
visibility: hidden
}

.css-1lsmgbg.egzxvld0
{
visibility: hidden
}
            
.css-1rf5dw.egzxvld0
{
visibility: hidden
}

</style>
""",unsafe_allow_html=True)


#-------------------- Databasase Configuration ---------------------------------------------------
# configuration key
firebaseConfig = {

  'apiKey': "AIzaSyDHVlAxccKX8inlTr99daVpnWrcLRVHsak",

  'authDomain': "scribosense.firebaseapp.com",

  'projectId': "scribosense",

  'storageBucket': "scribosense.appspot.com",

  'messagingSenderId': "161441601616",

  'appId': "1:161441601616:web:d44807a70d79bc3991d955",

  'databaseURL': "https://scribosense-default-rtdb.asia-southeast1.firebasedatabase.app/"

}

# firebase authentication
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# database 
db = firebase.database()

# -------------------------------------------------------------------------------------------------

st.markdown("<h1 style='text-align: center; font-size: 60px'>ScriboSense</h1>",unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center;'>The ultimate AI summary evaluater</h6>",unsafe_allow_html=True)

#---------------- For registered users ------------------------------------------------
# only registered users will see these functionalities

def registered_user():

    selected = option_menu(
        menu_title=None,
        options=["Summarize","Dashboard"],
        icons=["vector-pen","bar-chart-fill"],
        default_index=0,
        orientation="horizontal",
    )
    st.write("---")
    if selected == "Summarize":
        show_summarize_page()
    if selected == "Dashboard":
        show_dashboard()
    
    

#---------------------- for guest users-------------------------------------------------------
# guest users will see only these functonalities

def guest_user():
    st.write("---")
    show_summarize_page()
#------------------------------------------------------------------------------------------


if "userId" in st.session_state:
    registered_user()
    logout_btn=st.sidebar.button("Log Out")

    if logout_btn:
        # logout()
        del st.session_state["userId"]
        st.sidebar.success("Logged Out successfully!")
        st.experimental_rerun()
else:
    guest_user()


# if selected == "Login":
#     show_login_page()
# if selected == "Statistics":
#     st.write("Statistics page")