# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 19:03:00 2022

@author: Natalia

3) Del dataset anterior realice en PYTHON, tres algoritmos de preprocesamiento.
"""

import pandas as pd
import sklearn as skl
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
data = pd.read_csv("Maternal Health Risk Data Set.csv")
#print(data)

print('Normalizacion');
min_max_scaler = MinMaxScaler()
BS_values = data[['BS']]
scaled_values = min_max_scaler.fit(BS_values)
print(min_max_scaler.transform(BS_values)[0:50])

print('Estandarizacion');
standard_scaler = StandardScaler()
BodyTemp_values = data[['BodyTemp']]
scaled_values1 = standard_scaler.fit(BodyTemp_values)
print(standard_scaler.transform(BodyTemp_values)[0:50])

print('Discretizacion');
prepro = KBinsDiscretizer(n_bins=6,encode='ordinal' ,strategy='uniform')
SystolicBP_values = data[['SystolicBP']]
X2=prepro.fit_transform(SystolicBP_values)
print(prepro.transform(SystolicBP_values)[0:50])
