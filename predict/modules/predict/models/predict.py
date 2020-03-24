import os
import warnings
import pandas as pd
import matplotlib 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pickle
import requests
from  tqdm import tqdm
from datetime import datetime
from datetime import date
from sklearn.model_selection import KFold # gerador de data sets de treino e de teste para o modelo de regressão
from sklearn import linear_model # regressor do sklearn
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from scipy import stats
warnings.filterwarnings('ignore')
sns.set()

class Predict:

    def __init__(self, context):
        self.logging = context.logging

    def apply(self):
        # open a file, where you stored the pickled data
        # pickle_path = "/home/daniel/Documentos/stGobain/workspace/python/dev/"
        # file = open("/tmp/teste", 'wb')
        # file.write(requests.get('http://0.0.0.0:5002/models_files/esmat-model1').get_data())
        # file.close()
        #file = open(pickle_path + 'important', 'rb')
        file = open("tmp/modelfile/important", 'rb')

        # load information to that file
        lr = pickle.load(file)

        # close the file
        file.close()

        #####################################
        # Estudar como trazer X
        #####################################
        X = pd.read_csv('X.csv', encoding='utf-8')
        del X['Unnamed: 0']
        
        df_report = {}
        df_report['PREDICTION'] = lr.predict(X).tolist()
        df_report['PROBABILITY'] = [lr.predict_proba(X)[i][1] for i in range(lr.predict_proba(X).shape[0])]
        df_report['PERCENTIL(%)'] = [round(stats.percentileofscore(df_report['PROBABILITY'], a, 'weak'),2) for a in df_report['PROBABILITY']]

        return {"response":df_report}
            
   