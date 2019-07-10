#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 22:22:25 2019

@author: fabio
"""
#------------------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from scipy.stats import randint


#------------------------------------------------------------------------------
#Preparação dos dados
df = pd.read_csv('adult.data', header=None)
features = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','target']
df.columns = features
del features

df['class'] = df['target'].replace([' <=50K',' >50K'],[1,0])
df['workclass2'] = df['workclass'].replace(df['workclass'].unique(),range(len(df['workclass'].unique())))
df['education2'] = df['education'].replace(df['education'].unique(),range(len(df['education'].unique())))
df['marital-status2'] = df['marital-status'].replace(df['marital-status'].unique(),range(len(df['marital-status'].unique())))
df['occupation2'] = df['occupation'].replace(df['occupation'].unique(),range(len(df['occupation'].unique())))
df['relationship2'] = df['relationship'].replace(df['relationship'].unique(),range(len(df['relationship'].unique())))
df['race2'] = df['race'].replace(df['race'].unique(),range(len(df['race'].unique())))
df['sex2'] = df['sex'].replace(df['sex'].unique(),range(len(df['sex'].unique())))
df['native_country2'] = df['native_country'].replace(df['native_country'].unique(),range(len(df['native_country'].unique())))

#Retirando um amostra dos dados para diminuir o tempo de processamento
df_sample = df.sample(frac = 0.2, random_state = 132)


#------------------------------------------------------------------------------
#Separando os dados em treinamento e teste
features = ['age','education-num','fnlwgt','workclass2', 'education2', 'marital-status2', 'occupation2','relationship2', 'race2', 'sex2', 'native_country2']

X_train, X_test, y_train, y_test = train_test_split(np.array(df_sample[features]), 
                                                    np.array(df_sample['class']),
                                                    test_size=0.3, 
                                                    random_state=132)
del features


#------------------------------------------------------------------------------
#Treinando o modelo

parametros = {'n_estimators':[150, 200, 250],
              'criterion':['gini','entropy'],
              'max_depth':[3,4]}

busca = RandomizedSearchCV(estimator = RandomForestClassifier(),
                           param_distributions = parametros,
                           cv = KFold(n_splits=5))

busca.fit(X_train, y_train)

#Avaliacao do score via cross validate
cross_val_score(busca, X_test, y_test, cv = KFold(n_splits=5), scoring='precision')
#score = cross_val_score(busca, X_test, y_test, cv = KFold(n_splits=5), scoring='precision')
#score

#Melhor estimador
print(busca.best_estimator_)



#------------------------------------------------------------------------------
# Explorando o espaço de hiperparâmetros do algoritmo
parametros = {'n_estimators':randint(100, 300),
              'criterion':['gini','entropy'],
              'max_depth':randint(3, 8)}

busca = RandomizedSearchCV(estimator = RandomForestClassifier(),
                           param_distributions = parametros,
                           n_iter=4,
                           cv = KFold(n_splits=5, shuffle=True))

busca.fit(X_train, y_train)

# Avaliação do Score
cross_val_score(busca, X_test, y_test, cv = KFold(n_splits=5), scoring='precision')
print(busca.best_estimator_)



#------------------------------------------------------------------------------
#Avaliando os resultados
resultados = pd.DataFrame(busca.cv_results_)

resultados = resultados.sort_values("mean_test_score", ascending=False)
for indice, linha in resultados.iterrows():
    print("%.3f %.3f %s" % (linha.mean_test_score, linha.std_test_score, linha.params))
    






