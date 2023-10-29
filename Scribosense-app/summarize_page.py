import streamlit as st
from transformers import BertTokenizer
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import joblib
import torch

from predict_page import predict_content
from predict_page import predict_wording
import pyrebase


#---------------------------------------Database configurations --------------------------------------
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


#-----------------------------------------------------------------------------------------------------------

def scale_function(content,wording):

    # content scaling
    if content < -2:
        scaled_content = 10
    elif 5 < content:
        scaled_content = 95
    else:
        scaled_content = 10+80*(content+2)/7

    # wording scaling
    if wording < -2:
        scaled_wording = 10
    elif 5 < wording:
        scaled_wording = 95
    else:
        scaled_wording = 10+80*(wording+2)/7

    # overall percentage
    overall_percentage = (scaled_content*60/100)+(scaled_wording*40/100)

    # performance level
    if 0 <= overall_percentage < 30:
        performance = "Poor"
    elif 30 <= overall_percentage < 50:
        performance = "Below Average"
    elif 50 <= overall_percentage < 70:
        performance = "Average"
    elif 70 <= overall_percentage < 85:
        performance = "Good"
    elif 85 <= overall_percentage:
        performance = "Excellent"
    
    return scaled_content,scaled_wording,overall_percentage,performance


def add_summary_toDB(UID,title,prompt,text,summary,content_score,wording_score,scaled_content,scaled_wording,total_score,performance):
    if bool(db.child(UID).child("Submissions").get().val()):
        submissionID = len(db.child(UID).child("Submissions").get().val())
        # st.write(submissionID)
        # submissionID += 1
    else:
        submissionID = 0

    db.child(UID).child("Submissions").child(submissionID).child("title").set(title)
    db.child(UID).child("Submissions").child(submissionID).child("prompt").set(prompt)
    db.child(UID).child("Submissions").child(submissionID).child("text").set(text)
    db.child(UID).child("Submissions").child(submissionID).child("summary").set(summary)
    db.child(UID).child("Submissions").child(submissionID).child("content").set(content_score)
    db.child(UID).child("Submissions").child(submissionID).child("wording").set(wording_score)
    db.child(UID).child("Submissions").child(submissionID).child("content(scaled)").set(scaled_content)
    db.child(UID).child("Submissions").child(submissionID).child("wording(scaled)").set(scaled_wording)
    db.child(UID).child("Submissions").child(submissionID).child("total").set(total_score)
    db.child(UID).child("Submissions").child(submissionID).child("performance").set(performance)

def avg_score(UID):
    if "Submissions" in db.child(UID).get().val():
        user_submissions = db.child(UID).child("Submissions").get().val()
        total=0

        for i in range(len(user_submissions)):
            total += user_submissions[i]["total"]
        average = total/len(user_submissions)

        db.child(UID).child("Average").set(average)


def display_result(content_score,wording_score,performance):
    content_score = round(content_score,2)
    wording_score = round(wording_score,2)
    col1,col2,col3=st.columns(3)
    col1.metric("Content",content_score,delta=None)
    col2.metric("Wording",wording_score,delta=None)
    col3.metric("Performance",performance,delta=None)

def show_summarize_page():
    
    title = st.text_input("Enter the Title")
    prompt = st.text_input("Enter the Prompt")
    text = st.text_area("Enter the Text")
    summary = st.text_area("Enter your summary")
    clicked = st.button("submit")
    st.write("---")
    if clicked:
        # input title,prompt,text,summary to the model
        # get predicted values
        content_score = predict_content(prompt,title,text,summary)
        wording_score = predict_wording(prompt,title,text,summary,content_score)
        # wording_score = predict_wording(summary)

        scaled_content,scaled_wording,total_score,performance = scale_function(content_score,wording_score)
        display_result(scaled_content,scaled_wording,performance) # replace the values with content and wording scores predicted by ML model
        if "userId" in st.session_state:
            UID = st.session_state["userId"]
            add_summary_toDB(UID,title,prompt,text,summary,content_score,wording_score,scaled_content,scaled_wording,total_score,performance)
            avg_score(UID)

        # grammar_errors(summary)

        

        


#----------------------------------------------------------------------------------------------------------------

# st.write("Hi")

# summary = "the diffrent social class were like the diffrent part of the pyramid aka the govern if you were in the high class you are at the top of the pyramid lower class your the bottom of the pyramid or the base"

# wording_score = predict_wording(summary)
# st.write(wording_score)

