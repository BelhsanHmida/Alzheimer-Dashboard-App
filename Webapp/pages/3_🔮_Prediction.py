import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.neural_network import MLPClassifier



st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")

# Load your data
data = pd.read_csv(r'C:\Users\hp\Desktop\Alzheimer Dashboard\data\ADNI_Training_Q3_APOE_CollectionADNI1Complete 1Yr 1.5T_July22.2014.csv')
data = data.dropna()

X = data
Y = data['DX.bl']

remove_columns = list(X.columns)[0:9]
remove_columns.append('Dx Codes for Submission')
X = X.drop(remove_columns, axis=1)

features = list(X.columns)
numerical_vars = ['AGE', 'MMSE', 'PTEDUCAT']
cat_vars = list(set(features) - set(numerical_vars))
for var in cat_vars:
    one_hot_df = pd.get_dummies(X[var], prefix=var)
    X = pd.concat([X, one_hot_df], axis=1)
    X.drop(var, axis=1, inplace=True)
    
def normalize(X):
    X = np.array(X)
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    normalized_X = (X - means) / stds
    return normalized_X 

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=123)

# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(solver='lbfgs', penalty='l2', max_iter=1000000, multi_class='multinomial'),
    'Random Forest Classifier': RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                                    criterion='gini', max_depth=None, max_features='sqrt',
                                    max_leaf_nodes=None, max_samples=None,
                                    min_impurity_decrease=0.0, min_samples_leaf=1,
                                    min_samples_split=2, min_weight_fraction_leaf=0.0,
                                    n_estimators=100, n_jobs=-1, oob_score=False,
                                    random_state=123, verbose=0, warm_start=False),
    'Ridge Classifier': RidgeClassifier(alpha=1.0, copy_X=True, fit_intercept=True, random_state=123, solver='auto', tol=0.0001),
    'Gradient Boosting Classifier': GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                                    learning_rate=0.1, loss='log_loss', max_depth=3,
                                    max_features=None, max_leaf_nodes=None,
                                    min_impurity_decrease=0.0, min_samples_leaf=1,
                                    min_samples_split=2, min_weight_fraction_leaf=0.0,
                                    n_estimators=100, n_iter_no_change=None,
                                    random_state=123, subsample=1.0, tol=0.0001,
                                    validation_fraction=0.1, verbose=0,
                                    warm_start=False),
    'Multi-layer Perceptron Classifier': MLPClassifier(hidden_layer_sizes=(15, 10), alpha=3, learning_rate='adaptive', max_iter=100000)
}

# Sidebar
st.sidebar.title('Model Selection ')
selected_model = st.sidebar.selectbox('Choose a model', list(classifiers.keys()))

# Model training and evaluation
selected_clf = classifiers[selected_model]
selected_clf.fit(X_train, y_train)
y_pred = selected_clf.predict(X_test)

# Display results
st.title('Alzheimer\'s Disease Prediction')
st.write('-------')
st.write('## 🔍 Model Evaluation :', selected_model)

st.write('#### Evaluation Metrics')

# Define evaluation metrics
evaluation_metrics = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Value': [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, average='weighted'),
        recall_score(y_test, y_pred, average='weighted'),
        f1_score(y_test, y_pred, average='weighted')
    ]
}

# Create a DataFrame from the evaluation metrics dictionary
metrics_df = pd.DataFrame(evaluation_metrics)
red_color = '#ff0000'  # Red color
green_color = '#00ff00' 

# Define function to apply colors
def apply_color(value):
    trend_float = float(value)
    if trend_float > 0.5:
        return f'background-color: {green_color}; color: black'  # Green color
    elif trend_float < 0.5:
        return f'background-color: {red_color}; color: black'  # Red color
    else:
        return ''

# Apply color formatting to the 'Value' column
styled_metrics_df = metrics_df.style.applymap(apply_color, subset=['Value'])

 
# Display the styled DataFrame
st.write(styled_metrics_df)
 


import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(conf_matrix, classes):
    plt.figure(figsize=(11, 3))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Convert Matplotlib plot to Streamlit
    st.pyplot()


# Confusion matrix
st.write('### Confusion Matrix')
conf_matrix = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_matrix, selected_clf.classes_)