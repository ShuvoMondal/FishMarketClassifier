import streamlit as st
# loading the saved model
import pickle
loaded_model = pickle.load(open('FishClassifier.sav', 'rb')) 

# do same stuff...input data, reshape then predict
# for prediction instead of knn2 use loaded_model.predict()
st.write(loaded_model.predict([[600,17,18,20.3,7,2]]))