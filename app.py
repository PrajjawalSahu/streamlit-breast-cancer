#importing packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

#loaded the inbuilt breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

#Split the dataset into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Create a pandas DataFrame from the training data:
df_train = pd.DataFrame(X_train, columns=data.feature_names)
df_train['target'] = y_train

# st.dataframe(df_train)

# Select a smaller subset of the data
df_small = df_train.sample(frac=0.1)

# Use seaborn to visualize the relationship between the features and the target:
# sns.pairplot(df_small, hue='target')

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

st.title('Breast Cancer Classification')

if st.checkbox('Show data summary'):
    st.write(df_small.describe())

if st.checkbox('Show dataset'):
    st.write(df_small)

if st.checkbox('Show plot'):
    fig = sns.pairplot(df_small, hue='target', vars=['mean radius', 'mean perimeter', 'mean area', 'mean concavity'])
    # plt.savefig('plot3.pdf', format='pdf')  # working
    st.pyplot(fig)


if st.checkbox('Show confusion matrix'):
    st.write(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

if st.checkbox('Show accuracy'):
    st.write(clf.score(X_test, y_test))

if st.checkbox('Show feature importance'):
    st.write(clf.feature_importances_)

if st.checkbox('Show prediction'):
    st.write(clf.predict(X_test))

if st.checkbox('Show prediction probability'):
    st.write(clf.predict_proba(X_test))

