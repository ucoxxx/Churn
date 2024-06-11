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

# Memuat model yang disimpan
model = joblib.load('nn_Tuned.joblib')
