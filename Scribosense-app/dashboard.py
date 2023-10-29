import streamlit as st
import pyrebase
from streamlit_option_menu import option_menu
import plotly.express as px
import pandas as pd

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

#-------------------- User Dashboard ----------------------------------------------------------------------------------

def rank_function(UID):
    if "Submissions" in db.child(UID).get().val():
        average = db.child(UID).child("Average").get().val()

        if average < 30:
            return "Beginner 🧑"
        elif 30 <= average < 50:
            return "Apprentice 🧑‍🎓"
        elif 50 <= average < 70:
            return "Intermediate 🧑‍💼"
        elif 70 <= average < 85:
            return "Advanced 🫅"
        else:
            return "Expert 🧙‍♂️"
    else:
        return "Beginner 🧑"


def submission_history(UID):
    if "Submissions" in db.child(UID).get().val():
        st.dataframe(db.child(UID).child("Submissions").get().val())

def linechart(UID):
    if "Submissions" in db.child(UID).get().val():
        user_submissions = db.child(UID).child("Submissions").get().val()
        new_dic={}
        new_dic["content"]={}
        new_dic["wording"]={}

        for i in range(len(user_submissions)):
            
            index = "sub "+str(i)
            new_dic["content"][index]=user_submissions[i]["content(scaled)"]
            new_dic["wording"][index]=user_submissions[i]["wording(scaled)"]
        
        st.line_chart(data=new_dic,use_container_width=True)


def personal():
    
    if "userId" in st.session_state:
        UID = st.session_state["userId"]
        
        if "Submissions" in db.child(UID).get().val():
            Total_submissions=len(db.child(UID).child("Submissions").get().val())
            n=Total_submissions-1

            content_score = db.child(UID).child("Submissions").child(n).child("content(scaled)").get().val()
            wording_score = db.child(UID).child("Submissions").child(n).child("wording(scaled)").get().val()
            
            if Total_submissions >= 2:
                content_score_prev = db.child(UID).child("Submissions").child(n-1).child("content(scaled)").get().val()
                wording_score_prev = db.child(UID).child("Submissions").child(n-1).child("wording(scaled)").get().val()
            else:
                content_score_prev = 0
                wording_score_prev = 0
        else:
            content_score = 0
            wording_score = 0
            content_score_prev = 0
            wording_score_prev = 0
            Total_submissions = 0
        

        content_dif = content_score - content_score_prev
        wording_dif = wording_score - wording_score_prev

        # submission_history(UID)

        content_score=round(content_score,2)
        wording_score=round(wording_score,2)
        content_dif = round(content_dif,2)
        wording_dif = round(wording_dif,2)

        rank = rank_function(UID)

        col1,col2=st.columns(2)
        col1.metric("Total Submissions",Total_submissions,delta=None)
        col1.metric("Rank",rank,delta=None)
        col2.metric("Content Score",content_score,delta=content_dif)
        col2.metric("Wording Score",wording_score,delta=wording_dif)

        
        linechart(UID)
        submission_history(UID)


#-------------------------- Master Dashboard----------------------------------------------------

# def piechart():
#     poor = 0
#     below_avg=0
#     avg=0
#     good=0
#     excellent=0
#     for id in db.get().val().keys():
#         if "Submissions" in db.child(id).get().val():
#             user_submissions = db.child(id).child("Submissions").get().val()

#             for i in range(len(user_submissions)):
#                 if user_submissions[i]["performance"] == "Poor":
#                     poor += 1
#                 elif user_submissions[i]["performance"] == "Below Average":
#                     below_avg += 1
#                 elif user_submissions[i]["performance"] == "Average":
#                     avg += 1
#                 elif user_submissions[i]["performance"] == "Good":
#                     good += 1
#                 elif user_submissions[i]["performance"] == "Excellent":
#                     excellent += 1

#     labels = ["Poor","Below Average","Average","Good","Excellent"]
#     values = [poor,below_avg,avg,good,excellent]

#     fig = px.pie(names=labels,values=values,title="Performance of all students")
#     st.plotly_chart(fig, theme="streamlit", use_container_width=True)

def piechart():
    beginner = 0
    apprentice=0
    intermediate=0
    advanced=0
    expert=0
    for id in db.get().val().keys():
        if "Average" in db.child(id).get().val():
            average = db.child(id).child("Average").get().val()

            if average < 30:
                beginner += 1
            elif 30 <= average < 50:
                apprentice += 1
            elif 50 <= average < 70:
                intermediate += 1
            elif 70 <= average < 85:
                advanced += 1
            else:
                expert += 1
        else:
            beginner += 1
    
    labels = ["Beginner","Apprentice","Intermediate","Advanced","Expert"]
    values = [beginner,apprentice,intermediate,advanced,expert]

    fig = px.pie(names=labels,values=values,title="Ranks of all users")
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)


def leaderboard():
    name = []
    score = []
    for id in db.get().val().keys():
        if "Average" in db.child(id).get().val():
            score.append(db.child(id).child("Average").get().val())
            name.append(db.child(id).child("Name").get().val())
    data = {"Name":name,"Average_Score":score}

    df = pd.DataFrame(data)
    df = df.sort_values(by="Average_Score",ascending=False)

    st.write(df)


# function to get total submissions of all users
def master_tot_sub():

    total_submissions = 0
    for id in db.get().val().keys():
        if "Submissions" in db.child(id).get().val():
            total_submissions += len(db.child(id).child("Submissions").get().val())
    return total_submissions

# function to display master dashboard
def master():

    total_users = len(db.get().val())
    total_submissions = master_tot_sub()

    col1,col2 = st.columns(2)

    col1.metric("Number of Registered Users",total_users)
    col2.metric("Total Number of Submissions",total_submissions)

    with col1:
        st.header("Leaderboard")
        leaderboard()
        
    with col2:
        piechart()

#------------------------------------------------------------------------------------------------

def show_dashboard():
    selected = option_menu(
        menu_title=None,
        options=["User Dashboard","Master Dashboard"],
        icons=["graph-up","arrow-clockwise"],
        default_index=0,
        orientation="horizontal",
    )

    if selected == "User Dashboard":
        personal()
    if selected == "Master Dashboard":
        master()

