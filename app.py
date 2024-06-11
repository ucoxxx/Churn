import pandas as pd
import numpy as np
import sklearn.metrics as m
from sklearn.metrics import  roc_curve , auc , confusion_matrix, log_loss, brier_score_loss, f1_score
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
import sklearn.model_selection as cv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  roc_curve , auc , confusion_matrix
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, precision_score, recall_score
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import calibration_curve
import joblib
import warnings
import streamlit as st

# Memuat model yang disimpan
model = joblib.load('nn_Tuned.joblib')

def user_input_features():
    InternetService = st.selectbox('InternetService', options=['Fiber optic', 'No'])
    OnlineSecurity = st.selectbox('OnlineSecurity', options=['Yes', 'No'])
    OnlineBackup = st.selectbox('OnlineBackup', options=['Yes', 'No'])
    TechSupport = st.selectbox('TechSupport', options=['Yes', 'No'])
    StreamingTV = st.selectbox('StreamingTV', options=['No internet service', 'No'])
    Contract = st.selectbox('Contract', options=['Two year', 'No'])
    PaperlessBilling = st.selectbox('PaperlessBilling', options=['Yes', 'No'])
    PaymentMethod = st.selectbox('PaymentMethod', options=['Electronic check', 'No'])
    MonthlyCharges = st.number_input('MonthlyCharges', min_value=0, max_value=200000, value=0)
    
    # Buat dictionary untuk menyimpan input pengguna
    data = {
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges
    }

    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()

st.write(input_df)

categorical_preprocessor = OneHotEncoder(drop='first', handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

categorical_vars = input_df.select_dtypes(include=['O']).columns.tolist()

FS_preprocessor = ColumnTransformer(
    [
        ("one-hot-encoder", categorical_preprocessor, categorical_vars),
        ("standard-scaler", numerical_preprocessor)
    ]
)

FS_preprocessor.fit(input_df)

df_train = pd.DataFrame(FS_preprocessor.transform(input_df),columns=FS_preprocessor.get_feature_names_out())

Df_train_fs = df_train[best_features]

st.write(Df_train_fs.shape)

