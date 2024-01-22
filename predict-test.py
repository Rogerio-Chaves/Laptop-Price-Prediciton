# Import libraries.
from flask import request, jsonify

import json
import re
import requests
import pandas as pd

url = 'http://localhost:9696/predict'

df_test = pd.read_csv('data/test.csv')
y_test = df_test.price_euros.values
del df_test['price_euros']

for i in range(df_test.shape[0]):        
	json_file = df_test.iloc[i,:]
	json_file = dict(json_file)
	
	result = requests.post(url, json=json_file).json()
	pred_value = result['price']
	
	print(f'Value: ${y_test[i]:.2f} ------- Predict value: ${pred_value:.2f}')

