import pandas as pd
import streamlit as st
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier



########### Create a title
col1, col2 = st.columns([2, 1])
col1.title("Vehicle Loan Default Prediction 2018")
col2.image('image.png',width=200)

dic={
'DISBURSED_AMOUNT':st.number_input('DISBURSED AMOUNT (i.e 80000 if $80,000)',min_value=0,step=1),
'LTV':st.number_input('LOAN TO VALUE of the vehicle in percentage (i.e 75.89 if 75.89%)',min_value=0.00,format="%2.2f"),
'STATE_ID': st.selectbox(
    'STATE ID',
    ('01 - Andaman & Nicobar Is', 
     '02 - Andhra Pradesh',
     '03 - Arunachal Pradesh',
     '04 - Assam',
     '05 - Bihar',
     '06 - Chandigarh',
     '07 - Dadra & Nagar Haveli',
     '08 - Delhi',
     '09 - Goa, Daman & Diu',
     '10 - Gujarat',
     '11 - Haryana',
     '12 - Himachal Pradesh',
     '13 - Jammu & Kashmir',
     '14 - Kerala',
     '15 - Lakshadweep',    
     '16 - Madhya Pradesh',
     '17 - Maharashtra',
     '18 - Manipur',
     '19 - Meghalaya',
     '20 - Karnataka (Mysore)',
     '21 - Nagaland',
     '22 - Orissa')),
'PERFORM_CNS_SCORE':st.number_input('CREDIT SCORE',min_value=0,max_value=890,step=1),
'CREDIT_HISTORY_LENGTH':st.number_input('CREDIT HISTORY LENGTH in months (i.e 24 if 2 years)',min_value=0,step=1)}

state_dic={
     '01 - Andaman & Nicobar Is':1, 
     '02 - Andhra Pradesh':2,
     '03 - Arunachal Pradesh':3,
     '04 - Assam':4,
     '05 - Bihar':5,
     '06 - Chandigarh':6,
     '07 - Dadra & Nagar Haveli':7,
     '08 - Delhi':8,
     '09 - Goa, Daman & Diu':9,
     '10 - Gujarat':10,
     '11 - Haryana':11,
     '12 - Himachal Pradesh':12,
     '13 - Jammu & Kashmir':13,
     '14 - Kerala':14,
     '15 - Lakshadweep':15,
     '16 - Madhya Pradesh':16,
     '17 - Maharashtra':17,
     '18 - Manipur':18,
     '19 - Meghalaya':19,
     '20 - Karnataka (Mysore)':20,
     '21 - Nagaland':21,
     '22 - Orissa':22}

predict=st.button("Let's predict!")


for key, value in state_dic.items():
    if dic['STATE_ID'] == key:
        dic['STATE_ID'] = value

df=pd.DataFrame([dic])

X=pd.read_csv('X_train_5.csv')
y=pd.read_csv('y_train.csv')
rf = RandomForestClassifier(n_estimators=50, min_samples_leaf=5, max_depth=18).fit(X, y) 
if predict:
    if any(value == 0 for value in dic.values()):
        st.warning('Some inputs are zero. Please check and enter a positive integer.')
    else:
        prediction=rf.predict_proba(df)[:, 1]
        if prediction >=0.2:
            st.write('<p style="font-size:26px; color:red;">Likely to default!</p>',unsafe_allow_html=True)
        else:
            st.write('<p style="font-size:26px; color:green;">Not likely to default!</p>',unsafe_allow_html=True)


# st.write(STATE_ID)
# for key, value in state_dic.items():
#     if dic['STATE_ID'] == key:
#         dic['STATE_ID'] = value

# X_new=pd.DataFrame([dic])

# import base64
# import pickle

# with open('loan_default_new.pkl', 'rb') as f:
#     model_bytes = f.read()
# model_base64 = base64.b64encode(model_bytes).decode('utf-8')


# input=[DISBURSED_AMOUNT,LTV,STATE_ID,PERFORM_CNS_SCORE,CREDIT_HISTORY_LENGTH]   
# df = pd.DataFrame([input], columns=['DISBURSED_AMOUNT','LTV','STATE_ID','PERFORM_CNS_SCORE','CREDIT_HISTORY_LENGTH'])
# def change_value():
#     for index,values in df['STATE_ID'].items():
#         for key, value in state_dic.items():
#             if df.loc[index,'STATE_ID'] == key:
#                 df.loc[index,'STATE_ID'] = value

# change_value()

# print(df.values)
# my_array = np.array(input).reshape(1,len(input))
# st.write(df)
# st.write(type(my_array))


# model_base64 = st.secrets['model']['pickle_file']
# model_bytes = base64.b64decode(model_base64)
# model = pickle.loads(model_bytes)

# model = joblib.load('loan_default_new.pkl')

    
# with open('loan_default_new.pkl', 'rb') as file:
#     model = pickle.load(file)
# X_train=pd.read_csv('X_train.csv')
# y_train=pd.read_csv('y_train.csv')
# my_dict = {'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [7, 8, 9]}
# st.write(model.predict(X))
# rf = RandomForestClassifier(n_estimators=50, min_samples_leaf=5, max_depth=18).fit(X_train, y_train) 

# DISBURSED_AMOUNT=st.number_input('DISBURSED AMOUNT (i.e 80000 if $80,000)',min_value=0,step=1),
# LTV=st.number_input('LOAN TO VALUE of the vehicle in percentage (i.e 75.89 if 75.89%)',min_value=0.00,format="%2.2f"),
# STATE_ID=st.selectbox(
#     'STATE ID',
#     ('01 - Andaman & Nicobar Is', 
#      '02 - Andhra Pradesh',
#      '03 - Arunachal Pradesh',
#      '04 - Assam',
#      '05 - Bihar',
#      '06 - Chandigarh',
#      '07 - Dadra & Nagar Haveli',
#      '08 - Delhi',
#      '09 - Goa, Daman & Diu',
#      '10 - Gujarat',
#      '11 - Haryana',
#      '12 - Himachal Pradesh',
#      '13 - Jammu & Kashmir',
#      '14 - Kerala',
#      '15 - Lakshadweep',    
#      '16 - Madhya Pradesh',
#      '17 - Maharashtra',
#      '18 - Manipur',
#      '19 - Meghalaya',
#      '20 - Karnataka (Mysore)',
#      '21 - Nagaland',
#      '22 - Orissa')),
# PERFORM_CNS_SCORE=st.number_input('CREDIT SCORE',min_value=0,max_value=890,step=1),
# CREDIT_HISTORY_LENGTH=st.number_input('CREDIT HISTORY LENGTH in months (i.e 24 if 2 years)',min_value=0,step=1)