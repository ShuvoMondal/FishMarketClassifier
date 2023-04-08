import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Fish Market",
    page_icon="🐟",
    # '''🐟🎣🐠🦈'''
)

st.title("Fish Market Dataset")
st.markdown("![FISH GIF](https://media1.tenor.com/images/09a22b7bb2872243dfdd7bcdf88dc09e/tenor.gif?itemid=7957236)")
st.subheader("About")
st.write("This dataset is a record of 7 common different fish species in fish market sales. With this dataset, a predictive model can be performed using machine friendly data and estimate the species of fish can be predicted.")
st.markdown("Starter Notebook : [data-science-jobs-in-india](https://www.kaggle.com/datasets/aungpyaeap/fish-market)")
st.subheader("Attribute Information")
st.markdown('''
    - Species: Type of Fish
    - Weight: Weight of the Fish in grams
    - Length1: Vertical length of the Fish in centimeters
    - Length2: Diagonal length of the Fish in centimeters
    - Length3: Cross length of the Fish in centimeters
    - Height: Height of the Fish in centimeters
    - Width: Diagonal width of the Fish in centimeters
''')
fish_daigram = Image.open('./fishes/fish_size.jpg')
st.image(fish_daigram)
st.write("Based on the label from sciencedirect.com I think the below names should be used instead of :")
st.markdown('''
    - Length1 = Body_height
    - Length2 = Total_Length
    - Length3 = Diagonal_Length
''')


df = pd.read_csv("Fish.csv")

df.rename(columns = {'Length1':'Body_height', 
                     'Length2':'Total_Length',
                     'Length3':'Diagonal_Length'}, inplace = True)

st.dataframe(df, use_container_width=True)

st.subheader("Different Species")
st.write(df.value_counts("Species"))
