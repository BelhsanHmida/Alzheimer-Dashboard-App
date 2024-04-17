import streamlit as st 
import pandas as pd

st.set_page_config(page_title="Alzheimer's Variable Descriptions", page_icon="🏥", layout="wide")

 
st.title('🏥 Cognitive Compass')
st.write('-------')
st.markdown(
    """
## 🧠 **Introduction**

Alzheimer's disease is a complex neurodegenerative disorder that affects millions of people worldwide. Early detection and prediction of Alzheimer's can lead to better management and treatment outcomes. This prediction system utilizes a machine learning model trained on a dataset of relevant features to provide predictions about the likelihood of Alzheimer's disease.

## 🔍 **About Alzheimer's Disease**

Alzheimer's disease (AD) is a progressive neurodegenerative disease. Though best known for its role in declining memory function, symptoms also include difficulty thinking and reasoning, making judgments and decisions, and planning and performing familiar tasks. It may also cause alterations in personality and behavior. The cause of AD is not well understood. There is thought to be a significant hereditary component. For example, a variation of the APOE gene, APOE e4, increases the risk of Alzheimer's disease.

## 🎯 **Purpose of the Project**

The purpose of this project proposal is to develop a machine learning model for the early prediction of Alzheimer's disease. Alzheimer's disease is a devastating neurodegenerative disorder that affects millions of individuals worldwide. Early detection is crucial for better patient care and the development of potential interventions. This project aims to leverage machine learning techniques to create a predictive model that can identify individuals at risk of Alzheimer's disease based on relevant data.

## 💡 **Potential Impact**

The potential impact of this project on the issue of Alzheimer's disease is significant:

- Early prediction of Alzheimer's disease can lead to timely interventions, potentially slowing down the progression of the disease.
- Accurate prediction models can aid in identifying suitable candidates for clinical trials and research studies.
- Providing a tool for early prediction can raise awareness about Alzheimer's disease and encourage individuals to seek early medical evaluation.

The model will be trained on a dataset collected from [Alzheimer’s Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/). This dataset is a comprehensive collection of clinical, imaging, and genetic data from individuals with Alzheimer's disease.

---

"""
)

data = {
    'Variable': ['index', 'directory.id', 'Subject', 'RID', 'image.data.id', 'Modality', 'Visit', 'Acq.Date', 'DX.bl', 'EXAMDATE', 'AGE', 'PTGENDER', 'PTEDUCAT', 'PTETHCAT', 'PTRACCAT', 'APOE4', 'MMSE', 'imputed_genotype', 'APOE Genotype', 'Dx Codes for Submission'],
    'Description': ['A numerical index for each row in the dataframe.', 'A unique identifier for each row in the dataframe.', 'A unique identifier for each subject in the dataframe.', 'A numerical identifier for each subject in the dataframe.', 'A numerical identifier for each image in the dataframe.', 'The type of imaging used to collect the data (MRI).', 'The visit number associated with the data.', 'The date on which the data was acquired.', 'The diagnosis at baseline (AD, LMCI, or CN).', 'The date of the exam associated with the data.', 'The age of the subject at the time of the exam.', 'The gender of the subject.', 'The educational level of the subject.', 'The ethnicity of the subject (Hisp/Latino or Not Hisp/Latino).', 'The race of the subject (White).', 'The APOE4 genotype of the subject (0, 1, or 2).', 'The Mini-Mental State Examination score of the subject.', 'Whether or not the genotype was imputed (True or False).', 'The APOE genotype of the subject (3,3; 3,4; 4,3; 4,4).', 'The diagnosis code for submission (AD, MCI, or CN).']
}

df = pd.DataFrame(data)


st.title('Variable Descriptions')

# Display the table using st.table()
st.table(df)