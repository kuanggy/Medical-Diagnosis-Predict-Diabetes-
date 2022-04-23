import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd


st.title('Predict Diabetes Risk')
st.markdown('Using SVM model, Linear kernel. and Principal component analysis reduced to 7.')

@st.cache(persist=True)
def load_depen(model_path):
    column_names = ["pregnancies", "glucose", "bpressure", "skinfold", "insulin", "bmi", "pedigree", "age", "class"]
    df = pd.read_csv('data.csv', names=column_names)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)

    pca = PCA(n_components=7)
    X_train = pca.fit_transform(X_train)

    model = joblib.load(model_path)

    return sc, pca, model

sc, pca, model = load_depen('model.pkl')
col1, col2 = st.columns(2)
with col1:
    preg = st.selectbox('Pregnancies', range(1,17))
    glucose = st.number_input('Fasting Blood Sugar', 20., 500., step=0.01)
    bp = st.number_input('Blood Pressure (Diastolic)', 20., 500., step=0.01)
    skinfold = st.number_input('Skin Fold', 20., 500., step=0.01)

with col2:
    insulin = st.number_input('Insulin Level', 0., 500., step=0.01)
    bmi = st.number_input('BMI', 5., 100., step=0.01)
    pedigree = st.number_input('Pedigree', 0., 100., step=0.5)
    age = st.number_input('Age', 1., 150., step=0.1)

try:
    preg,glucose,bp,skinfold,insulin,bmi,pedigree,age = float(preg),float(glucose),float(bp),float(skinfold),float(insulin),float(bmi),float(pedigree),float(age)
except ValueError:
    print('Some input is not a number')

input_data = [[preg,glucose,bp,skinfold,insulin,bmi,pedigree,age]]
input_data = sc.transform(input_data)
input_data = pca.transform(input_data)
prediction = model.predict(input_data)
pred_proba = model.predict_proba(input_data)

if prediction == 0:
    result = 'This patient may have no DM.'
    prob = f'Confidence Rate: {round(pred_proba[0][np.argmax(pred_proba)], 3)}'
else:
    result = 'This patient may HAVE DM.'
    prob = f'Percent Confidence Rate: {round(pred_proba[0][np.argmax(pred_proba)], 3) * 100}%'

if st.button('Start Calculate'):
    st.write(result, prob)
else:
    st.write('No calculate result')