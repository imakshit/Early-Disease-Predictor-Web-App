import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Disease Prediction App
This app predicts **Diabetes Hypertension and Depression**!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    Age = st.sidebar.slider('Age', 1, 100, 50)
    Bmi = st.sidebar.slider('BMI',10 , 50, 30)
    Drinking = st.sidebar.slider('DRINKING ', 0, 1, 0)
    Exercise = st.sidebar.slider('EXERCISE PER WEEK', 1, 3, 1)
    Gender = st.sidebar.slider('GENDER (0:M , 1:F)', 0, 1, 0)
    Junk = st.sidebar.slider('JUNK PER WEEK', 1, 3, 2)
    Sleep = st.sidebar.slider('SLEEP SCORE', 1, 3, 1)
    Smoking = st.sidebar.slider('SMOKING', 0, 1, 0)
  
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

df = pd.read_csv('d_sih.csv')
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
a = ans[0]
b = ans[1]
c = ans[2]
if(a<20 and a>b and a>c):
    st.write('You have a low risk of Diabetes')
elif(b<20 and b>c and b>a):
    st.write("You have a low risk of HYPERTENSION")
elif(c<20 and c>a and c>b):
     st.write("You have a low risk of DEPRESSION!")           
elif(a>b and a>c):
     
     st.write("You have a high risk of DIABETES!")
                
elif (b>a and b>c):
     
     st.write("You have a high risk of HYPERTENSION")
else:
     
     st.write("You have a high risk of DEPRESSION!")







#target_names = ['Diabetes','Hypertension','Depression']

#st.subheader('Class labels and their corresponding index number')
#st.write(df.target_names)

#st.subheader('Prediction')
#st.write(df.target_names[prediction])
#st.write(prediction)
prediction_proba = classifier.score(X_test,y_test)

st.subheader('Model Accuracy')
st.write(prediction_proba)
