import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Disease Prediction App
This app predicts risk of **Depression, Diabetes and Hypertension**!
Results are based on Data analysis
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    Age = st.sidebar.slider('Age', 1, 100, 50)
    Bmi = st.sidebar.slider('BMI',10 , 50, 30)
    st.sidebar.markdown("BMI Calculator: <https://www.calculator.net/bmi-calculator.html>")
    #Drinking = st.sidebar.slider('DRINKING ', 0, 1, 0)
    Drinking = st.sidebar.radio( 'DRINKING (0:N, 1:Y)',options = [0,1], index=1)
    #Exercise = st.sidebar.slider('EXERCISE PER WEEK', 1, 3, 1)
    Exercise = st.sidebar.radio( 'EXERCISE PER WEEK',options = [1,2,3], index=1)
    Gender = st.sidebar.radio('GENDER (0:M , 1:F)', options = [0,1])
    Junk = st.sidebar.radio('JUNK PER WEEK', options = [1,2,3])
    Sleep = st.sidebar.radio('SLEEP SCORE', options= [1,2,3])
    Smoking = st.sidebar.radio('SMOKING (0:N , 1:Y)', options = [0,1])
  
    data = {'Age': Age,
            'Bmi': Bmi,
            'Drinking': Drinking,
            'Exercise': Exercise,
            'Gender': Gender,
            'Junk': Junk,
            'Sleep': Sleep,
            'Smoking': Smoking,
            }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('User Input parameters')
st.write(df_input)

df = pd.read_csv('dataset.csv')
X = df.iloc[:, 0:8].values
y = df.iloc[:, 8:11].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler as ss
sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


classifier= RandomForestRegressor(n_estimators = 300, random_state = 0)
classifier.fit(X_train,y_train)

prediction = classifier.predict(df_input)
#prediction_proba = classifier.score(X_test, y_test)
st.subheader('Prediction')

ans = prediction.flatten()
a = ans[0] #depression
b = ans[1] #diabetes
c = ans[2] #hypertension
if(a<50 and b<50 and c<50):
    st.write("You are completely fit and healthy")
    st.balloons()
elif(a>50 and a<70 and a>b and a>c):
    st.write('You have a low risk of DEPRESSION!')
elif(b>50 and b<70 and b>c and b>a):
    st.write("You have a low risk of DIABETES!")
elif(c>50 and c<70 and c>a and c>b):
     st.write("You have a low risk of HYPERTENSION!")           
elif(a>50 and a>b and a>c):
     
     st.write("You have a high risk of DEPRESSION!")
                
elif (b>50 and b>a and b>c):
     
     st.write("You have a high risk of DIABETES")
elif (c>50 and c>a and c>b):     
     st.write("You have a high risk of HEPERTENSION!")


     
     
prediction_proba = classifier.score(X_test,y_test)

st.subheader('Model Accuracy')
st.write(prediction_proba)
