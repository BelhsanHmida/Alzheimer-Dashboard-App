import numpy as np
import pandas as pd
 
import matplotlib.pyplot as plt
import joblib
import plotly.graph_objs as go
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Read the data
data = pd.read_csv(r'C:\Users\hp\Desktop\Alzheimer Dashboard\data\ADNI_Training_Q3_APOE_CollectionADNI1Complete 1Yr 1.5T_July22.2014.csv')

# Drop NA values
data = data.dropna()

# Define features and target
X = data.drop(columns=['DX.bl', 'Dx Codes for Submission'])
Y = data['DX.bl']

# Visualizations
st.set_page_config(page_title="Prediction", page_icon="📊", layout="wide")

# Sidebar
st.sidebar.title('Select Visualization')
visualization_option = st.sidebar.selectbox('Choose a visualization', 
                                             ['Scatter plot for Age vs Diagnosis', 
                                              'Gender and Ethnicity Counts', 
                                              'Relationship between APOE4 genotype and diagnosis',
                                              'Scatter plot for MMSE vs Diagnosis', 
                                              'Distribution of Imputed Genotype Across Different Diagnoses',
                                              'APOE Genotype vs Diagnosis',
                                              'AGE Distribution',
                                              'MMSE Distribution',
                                              'PTEDUCAT Distribution',
                                              'Diagnosis Labels Distribution'])

# Display selected visualization
if visualization_option == 'Scatter plot for Age vs Diagnosis':
    st.subheader('Scatter plot for Age vs Diagnosis')
    fig_age_diagnosis = px.scatter(data, x='AGE', y='DX.bl')
    st.plotly_chart(fig_age_diagnosis)

elif visualization_option == 'Gender and Ethnicity Counts':
    st.subheader('Gender and Ethnicity Counts')
    gender_ethnicity_counts = data.groupby(['DX.bl', 'PTGENDER', 'PTETHCAT']).size().reset_index(name='count')
    st.write(gender_ethnicity_counts)

elif visualization_option == 'Relationship between APOE4 genotype and diagnosis':
    st.subheader('Relationship between APOE4 genotype and diagnosis')
    fig_apoe4_diagnosis = px.histogram(data, x='APOE4', color='DX.bl', barmode='group')
    st.plotly_chart(fig_apoe4_diagnosis)

elif visualization_option == 'Scatter plot for MMSE vs Diagnosis':
    st.subheader('Scatter plot for MMSE vs Diagnosis')
    fig_mmse_diagnosis = px.scatter(data, x='MMSE', y='DX.bl')
    st.plotly_chart(fig_mmse_diagnosis)

elif visualization_option == 'Distribution of Imputed Genotype Across Different Diagnoses':
    st.subheader('Distribution of Imputed Genotype Across Different Diagnoses')
    counts = data.groupby(['DX.bl', 'imputed_genotype']).size().reset_index(name='counts')
    fig_imputed_genotype = px.bar(counts, x='DX.bl', y='counts', color='imputed_genotype', barmode='group')
    st.plotly_chart(fig_imputed_genotype)

elif visualization_option == 'APOE Genotype vs Diagnosis':
    st.subheader('APOE Genotype vs Diagnosis')
    fig_apoe_genotype = px.bar(data.groupby(['APOE Genotype', 'DX.bl']).size().reset_index(name='counts'), 
                               x='APOE Genotype', y='counts', color='DX.bl', barmode='group')
    st.plotly_chart(fig_apoe_genotype)

elif visualization_option == 'AGE Distribution':
    st.subheader('AGE Distribution')
    fig_age_dist = px.histogram(data, x='AGE', title='Age Distribution')
    st.plotly_chart(fig_age_dist)

elif visualization_option == 'MMSE Distribution':
    st.subheader('MMSE Distribution')
    fig_mmse_dist = px.histogram(data, x='MMSE', title='MMSE Distribution')
    st.plotly_chart(fig_mmse_dist)

elif visualization_option == 'PTEDUCAT Distribution':
    st.subheader('PTEDUCAT Distribution')
    fig_pteducat_dist = px.histogram(data, x='PTEDUCAT', title='PTEDUCAT Distribution')
    st.plotly_chart(fig_pteducat_dist)

elif visualization_option == 'Diagnosis Labels Distribution':
    st.subheader('Diagnosis Labels Distribution')
    fig_diagnosis_dist = px.bar(Y.value_counts(), title='Diagnosis Labels Distribution')
    st.plotly_chart(fig_diagnosis_dist)
