import streamlit as st
import pyrebase
from datetime import datetime
from streamlit_option_menu import option_menu

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

.css-14xtw13.e8zbici0
{
visibility: hidden
}
</style>
""",unsafe_allow_html=True)


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

def signup_function():
    name = st.text_input("Name")
    age = st.number_input(label="Age",min_value=8,max_value=19)
    email = st.text_input("Email Address")
    password = st.text_input("Password",type="password")
    acc_created = st.button("Create an Account")
    if acc_created:
        if len(password) < 6:
            st.error("Week password: Your passwrod should contain atleast 6 characters.")
        else:
            try:
                user = auth.create_user_with_email_and_password(email,password)
                add_user(user["localId"],name,age)
                st.success("Your account is created successfully!")
                st.balloons()
            except:
                st.error("An account for this email address already exists.")
        

def login_function():
    email = st.text_input("Email Address")
    password = st.text_input("Password",type="password")
    logged_in = st.button("Log in")

    if logged_in:
        try:
            user = auth.sign_in_with_email_and_password(email,password)
            st.header("Welcome!")
            # st.write(user["localId"])
            st.session_state["userId"] = user["localId"]
        except:
            st.error("Invalid login credentials.")

def add_user(UID,name,age):
    db.child(UID).child("User_ID").set(UID)
    db.child(UID).child("Name").set(name)
    db.child(UID).child("Age").set(age)
    


def show_login_page():

    selected = option_menu(
    menu_title=None,
    options=["Login","Create a new account"],
    icons=["person-fill","pencil-square"],
    default_index=0,
    orientation="horizontal",
    )
    st.write("---")

    # choice = st.selectbox('Login/Signup',['Login','Signup'])

    # if choice == 'Login':
    #     login_function()
    # if choice == 'Signup':
    #     signup_function()
    if selected == "Login":
        login_function()
    if selected == "Create a new account":
        signup_function()

def logout():
    # auth.current_user=None
    del st.session_state["userId"]
    st.sidebar.success("Logged Out successfully!")

#---------------------------------------------------------------------------------------------------------------
st.markdown("<h1 style='text-align: center; font-size: 60px'>ScriboSense</h1>",unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center;'>The ultimate AI summary evaluater</h6>",unsafe_allow_html=True)
st.write()
show_login_page()

if "userId" in st.session_state:
    logout_btn=st.sidebar.button("Log Out")

    if logout_btn:
        logout()
        st.experimental_rerun()

# st.write("this is login page")
