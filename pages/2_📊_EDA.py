import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px

df = pd.read_csv("Fish.csv")

st.subheader("Feilds Renaming")
df.rename(columns = {'Length1':'Body_height', 
                     'Length2':'Total_Length',
                     'Length3':'Diagonal_Length'}, inplace = True)

code = '''
df.rename(columns = {'Length1':'Body_height', 
                     'Length2':'Total_Length',
                     'Length3':'Diagonal_Length'}, inplace = True)
'''
st.code(code, language='python')

#data cleaning
st.subheader("Data cleaning")

df = df.drop_duplicates()
df = df.dropna()

code = '''
df = df.drop_duplicates()
df = df.dropna()'''

st.code(code, language='python')

st.subheader("Data of Different Species")
speciesOption = df['Species'].unique().tolist()

species = st.multiselect('Select Species',speciesOption,['Roach'])
speciesData = df[df['Species'].isin(species)]

st.subheader("Summarized of Different Species")
st.bar_chart(data=speciesData,x='Species',use_container_width=True)

st.subheader("Popular choice of fish in the market")
count_fig = px.bar(df,x='Species', color='Species')
st.write(count_fig)
st.write('''
This Analysis has been performed on the basis of the count of species that was available in the dataset.

Here, we are assuming that a higher count signifies the fact that the fish is readily available in the market. It also hints that it is popular among the people in that area.

Perch is the most popular type of fish here, followed by Bream.
''')

st.subheader("Average Weigth of fish in the market")
weight_fig = px.bar(df,y='Weight', color='Species')
st.write(weight_fig)
st.write('''
When we check the average weight of all the fish, we see that Bream, Whitefish, Perch and Pike are relatively heavy.

Another thing to notice here is that, Bream is a popular fish. Maybe people eat this fish during some occasion or celebration!.
''')

st.subheader("Average Hight of fish in the market")
hight_fig = px.bar(df,x='Height',color='Species')
st.write(hight_fig)

st.write('''
When we check the average height of all the fish, we see that Bream, Whitefish and Parkki are relatively long. Especially Bream.

Perch which is popular, has an average height out of all the fish. This can be due to the fact that it is consumed almost everytime by the people.

Therefore, Perch is for daily consumption whereas, Bream preferably on a special occasion. Okay, I might have to try this fish someday 😜.
''')





