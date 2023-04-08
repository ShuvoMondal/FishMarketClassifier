import streamlit as st
import pandas as pd
import joblib
from PIL import Image
# loading the saved model
import pickle
knn_clf = pickle.load(open('FishClassifier.sav', 'rb')) 

st.title("Fish Market Classification")
#Loading images
# setosa= Image.open('setosa.png')
# versicolor= Image.open('versicolor.png')
# virginica = Image.open('virginica.png')

def loadFish(fish):
    st.header(fish)
    img = Image.open('./fishes/{0}.jpg'.format(fish))
    st.image(img)

#Intializing
parameter_list=['Weight (grams)','Body height (cm)','Total length (cm)','Diagonal length (cm)','Height (cm)', 'Width (cm)']
min_max_value=[(1.0,2000.0),(1.0,100.0),(1.0,100.0),(1.0,100.0),(1.0,100.0),(1.0,100.0)]
parameter_input_values=[]
parameter_default_values=['600','17','18','20.3','7','2']
values=[]

#Display
for parameter,min_max, parameter_df in zip(parameter_list,min_max_value, parameter_default_values):
   values= st.slider(label=parameter, key=parameter,value=float(parameter_df), min_value=min_max[0], max_value=min_max[1], step=0.1)
   parameter_input_values.append(values)
 
input_variables=pd.DataFrame([parameter_input_values],dtype=float)
st.write('\n\n')

if st.button("Click Here to Classify"):
    print(input_variables)
    prediction = knn_clf.predict(input_variables)[0]
    loadFish(prediction)

# st.image(setosa) if prediction == 0 else st.image(versicolor)  if prediction == 1 else st.image(virginica)