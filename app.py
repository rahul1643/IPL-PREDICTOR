import streamlit as st
import pickle
import pandas as pd
import sklearn

st.title("IPL WIN PREDICTOR")
teams=['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities=['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah','Mohali', 'Bengaluru']

pipe=pickle.load(open('pipe.pkl','rb'))
col1,col2=st.columns(2)

with col1:
    batting_team=st.selectbox('Select the batting team',teams)
    

with col2:
    bowling_team=st.selectbox('Select the bowling team',teams)
    
selected_city=st.selectbox('Select the Host City',cities)
target=st.number_input('Enter the Target')

col3,col4,col5=st.columns(3)

with col3:
    Score=st.number_input('Enter The Score')
    

with col4:
    Overs=st.number_input('Enter The Overs')
    

with col5:
    Wickets=st.number_input('Enter The Wickets Fallen')
   
if st.button('Predict Probability'):
    runs_left=target-Score
    balls_left=120-(Overs*6)
    wickets=10-Wickets
    crr=Score/Overs
    rrr=(runs_left*6)/balls_left
    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")



                   