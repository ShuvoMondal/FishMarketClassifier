import streamlit as st

import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("Fish.csv")
df.rename(columns = {'Length1':'Body_height', 
                     'Length2':'Total_Length',
                     'Length3':'Diagonal_Length'}, inplace = True)
df = df.drop_duplicates()
df = df.dropna()

st.header("Clustering")
st.write('''
As you enjoy a hike in the mountains, you stumble upon a plant you have never seen before. You look around and you notice a few more. They are not identical, yet they are sufficiently similar for you to know that they most likely belong to the same species (or atleast the same genus). You may need a botanist to tell you what species that is, but you certainly don't need an expert to identify all similar looking objects. This is called Clustering. It is the task of identifying similar instances and assign them to clusters, or groups of similar instances.
Clustering can be used in a wide variety of applications, including:
''')
st.markdown('''
    - **For Customer Segmentation:** You can cluster customers based on theirpurchases and their activity on your website. This is helpful to understand who your customers are and what they need.
    - **As a Dimensionality Reduction Technique:** Once a dataset has been clustered, it is usually possible to measure each instance's affinity with each cluster (affinity is how well an instance fits into a cluster).
    - **For Anomaly Detection:** Any instance that has a low affinity to a cluster is likely to be an anomaly. For example, if you have clustered the users of your website based on their behavior, you can detect users with unusual number of requests per second. Anomaly detection is particularly helpful in detecting defects in manufacturing, or for fraud detection.
''')

st.subheader('Cluster of fish arranged according to height and weight ')      
speciesOption = df['Species'].unique().tolist()
species = st.multiselect('Select Species',speciesOption,['Roach','Bream','Parkki'])
speciesData = df[df['Species'].isin(species)]

cluster_hvw = px.scatter(speciesData, x="Height", y="Weight", size='Width', color="Species")
st.write(cluster_hvw)

st.subheader('An Overview of K-Nearest Neighbors')   
st.write('''
The kNN algorithm can be considered a voting system, where the majority class label determines the class label of a new data point among its nearest ‘k’ (where k is an integer) neighbors in the feature space. Imagine a small village with a few hundred residents, and you must decide which political party you should vote for. To do this, you might go to your nearest neighbors and ask which political party they support. If the majority of your’ k’ nearest neighbors support party A, then you would most likely also vote for party A. This is similar to how the kNN algorithm works, where the majority class label determines the class label of a new data point among its k nearest neighbors.

Let's take a deeper look with another example. Imagine you have data about fruit, specifically grapes and pears. You have a score for how round the fruit is and the diameter. You decide to plot these on a graph. If someone hands you a new fruit, you could plot this on the graph too, then measure the distance to k (a number) nearest points to decide what fruit it is. In the example below, if we choose to measure three points, we can say the three nearest points are pears, so I’m 100% sure this is a pear. If we choose to measure the four nearest points, three are pears while one is a grape, so we would say we are 75% sure this is a pear. We’ll cover how to find the best value for k and the different ways to measure distance later in this article.
''')

st.markdown("![Cluster Example](https://res.cloudinary.com/dyd911kmh/image/upload/v1676909140/image2_26000761c3.png)")

st.subheader("The Dataset")
st.dataframe(df, use_container_width=True)

st.header("k-Nearest Neighbors Workflow")
st.markdown('''
However, as our data is pretty clean, we won’t carry out every step. We will do the following:
    - Feature engineering
    - Spliting the data
    - Train the model
    - Hyperparameter tuning
    - Assess model performance

''')

st.subheader("Visualize the Data")
st.write("Let’s start by visualizing our data using plotly; we can plot our two features in a scatterplot.")

code = '''
cluster_hvw = px.scatter(df, x="Height", y="Weight", size='Width', color="Species")
'''
st.code(code, language='python')

cluster_hvw = px.scatter(df, x="Height", y="Weight", size='Width', color="Species")
st.write(cluster_hvw)

st.markdown("Looking at the scatter plot of weight vs. height for the fish market dataset, we can observe the following:")
st.markdown('''
    - There is a positive correlation between weight and height, which means that as the weight of the fish increases, its height also tends to increase.
    - However, the relationship is not perfectly linear, and there is some variation in the data.
    - There are some outliers in the upper right corner of the plot, indicating that there are a few fish that are significantly larger and heavier than the rest of the population.
    - The scatter plot provides a good visual representation of the relationship between weight and height and can help us identify any patterns or trends in the data.

''')

st.subheader("Normalizing & Splitting the Data")
st.markdown('''
When training any machine learning model, it is important to split the data into training and test data. The training data is used to fit the model. The algorithm uses the training data to learn the relationship between the features and the target. It tries to find a pattern in the training data that can be used to make predictions on new, unseen data. The test data is used to evaluate the performance of the model. The model is tested on the test data by using it to make predictions and comparing these predictions to the actual target values. 

When training a kNN classifier, it's essential to normalize the features. This is because kNN measures the distance between points. The default is to use the Euclidean Distance, which is  the square root of the sum of the squared differences between two points. 

We should normalize the data after splitting it into training and test sets. This is to prevent ‘data leakage’ as the normalization would give the model additional information about the test set if we normalized all the data at once.

The following code splits the data into train/test splits, then normalizes using scikit-learn’s standard scaler. We first call .fit_transform() on the training data, which fits our scaler to the mean and standard deviation of the training data. We can then apply this to the test data by calling .transform(), which uses the previously learned values.
''')
            
code = '''
#seperate the target from feature
y=df['Species'] 
X = df.drop(['Species'], axis=1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

'''
st.code(code, language='python')

y=df['Species'] 
X = df.drop(['Species'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st.subheader("Fitting and Evaluating the Model")
st.markdown('''
We are now ready to train the model. For this, we’ll use a fixed value of 3 for k, but we’ll need to optimize this later on. We first create an instance of the kNN model, then fit this to our training data. We pass both the features and the target variable, so the model can learn.
''')
code = '''
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)
'''
st.code(code, language='python')

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)

st.markdown("The model is now trained! We can make predictions on the test dataset, which we can use later to score the model.")
code = '''
predictions = knn.predict(X_test)
'''
st.code(code, language='python')

predictions = knn.predict(X_test)

code = '''
#model Evaluation
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
'''
st.code(code, language='python')

st.write(confusion_matrix(y_test, predictions))
st.write(classification_report(y_test, predictions))

st.subheader("Using Cross Validation to Get the Best Value of k")
st.markdown('''
Unfortunately, there is no magic way to find the best value for k. We have to loop through many different values, then use our best judgment.

In the below code, we select a range of values for k and create an empty list to store our results. We use cross-validation to find the accuracy scores, which means we don’t need to create a training and test split, but we do need to scale our data. We then loop over the values and add the scores to our list.
''')
code = '''
# test the algo for multiple K-values
# for this we will find error values for each K from 1 to 31
error_value = []
for i in range(1,31):
    knn= KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    prediction_i = knn.predict(X_test)
    error_value.append(np.mean(prediction_i != y_test))

print(error_value)
print(min(error_value))
print(error_value.index(min(error_value)))  
# index of min value +1 will be the best K value
'''
st.code(code, language='python')     

# test the algo for multiple K-values
# for this we will find error values for each K from 1 to 31
error_value = []
for i in range(1,31):
    knn= KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    prediction_i = knn.predict(X_test)
    error_value.append(np.mean(prediction_i != y_test))

st.write(error_value)
st.write(min(error_value))
st.write(error_value.index(min(error_value)))  
# index of min value +1 will be the best K value

code = '''
fig = plt.figure()
plt.plot(range(1,31),error_value)
plt.xlabel("K-value")
plt.ylabel("Error Value")
'''
st.code(code, language='python')  

fig = plt.figure()
plt.plot(range(1,31),error_value)
plt.xlabel("K-value")
plt.ylabel("Error Value")

st.pyplot(fig)

st.subheader("More Evaluation Metrics")
st.markdown(''''
We can now train our model using the best k value using the code below. then evaluate with accuracy, precision, and recall (note your results may differ due to randomization)
''')
            
code = '''
# Re-evluate the model with K=2
knn2 = KNeighborsClassifier(n_neighbors = 2)
knn2.fit(X_train,y_train)
pred = knn2.predict(X_test)

print(classification_report(y_test, pred))
# print(knn2.predict([[600,17,18,20.3,7,2]]))
'''
st.code(code, language='python')  

# Re-evluate the model with K=2
knn2 = KNeighborsClassifier(n_neighbors = 2)
knn2.fit(X_train,y_train)
pred = knn2.predict(X_test)

st.write(classification_report(y_test, pred))
print(knn2.predict([[600,17,18,20.3,7,2]]))

st.subheader("Saving the model")
code = '''
# saving the model
import pickle
pickle.dump(knn2, open('FishClassifier.sav', 'wb')) 
'''
st.code(code, language='python')  
# saving the model
# import pickle
# pickle.dump(knn2, open('FishClassifier.sav', 'wb')) 