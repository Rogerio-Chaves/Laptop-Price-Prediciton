# Import libraries.
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import re
from sklearn.preprocessing import LabelEncoder

import numpy as np
import os
import pandas as pd
import pickle


def to_numeric(value):
    number = re.search(r'[0-9]+([.][0-9]+)?', value).group()
    try:
        if '.' in number:
            number = float(number)
            return number
        else:
            number = int(number)
            return number
    except ValueError:
        print('Isn\'t a number!')

# Load the dataset.
df = pd.read_csv('data/laptop_price.csv', encoding='cp437')
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Load relevant columns for the model.
with open('model/relevant_columns.bin', 'rb') as f:
    relevant_columns = pickle.load(f)
    f.close()
# Select relevant columns
df = df[relevant_columns]

# Label encoding
cpu_encoder = LabelEncoder()
gpu_encoder = LabelEncoder()
product_encoder = LabelEncoder()
memory_encoder = LabelEncoder()
typename_encoder = LabelEncoder()
resolution_encoder = LabelEncoder()

cpu_encoder.fit(df['cpu'])
gpu_encoder.fit(df['gpu'])
product_encoder.fit(df['product'])
memory_encoder.fit(df['memory'])
typename_encoder.fit(df['typename'])
resolution_encoder.fit(df['screenresolution'])

# Train and test split.
df_train, df_test = train_test_split(df, test_size=0.20, random_state=1)

# Save the test dataset.
DATASETS_DIR = 'data'
if not os.path.exists(DATASETS_DIR):
    os.makedirs(DATASETS_DIR)
    print('Datasets directory was created!')

df_test.to_csv(f'{DATASETS_DIR}/test.csv', index=False)

# Change string values to numeric on training dataset.
df_train.ram = df_train.ram.apply(to_numeric)
df_train.weight = df_train.weight.apply(to_numeric)

# Apply label encoding on training dataset
df_train['cpu'] = cpu_encoder.transform(df_train['cpu'])
df_train['gpu'] = gpu_encoder.transform(df_train['gpu'])
df_train['product'] = product_encoder.transform(df_train['product'])
df_train['memory'] = memory_encoder.transform(df_train['memory'])
df_train['typename'] = typename_encoder.transform(df_train['typename'])
df_train['screenresolution'] = resolution_encoder.transform(df_train['screenresolution'])

# Split predictors and target on training dataset.
y_train = df_train.price_euros.values
del df_train['price_euros']

# Dict Vectorizer
dicts = df_train.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(dicts)

# Training the model
rf_model = RandomForestRegressor(criterion='poisson', min_samples_leaf=1, min_samples_split=2, n_estimators=100)
rf_model.fit(X_train, y_train)

# Change string values to numeric on testing dataset.
df_test.ram = df_test.ram.apply(to_numeric)
df_test.weight = df_test.weight.apply(to_numeric)

# Apply label encoding on testing dataset
df_test['cpu'] = cpu_encoder.transform(df_test['cpu'])
df_test['gpu'] = gpu_encoder.transform(df_test['gpu'])
df_test['product'] = product_encoder.transform(df_test['product'])
df_test['memory'] = memory_encoder.transform(df_test['memory'])
df_test['typename'] = typename_encoder.transform(df_test['typename'])
df_test['screenresolution'] = resolution_encoder.transform(df_test['screenresolution'])

# Split predictors and target on testing dataset.
y_test = df_test.price_euros.values
del df_test['price_euros']
X_test = df_test.values
X_test.shape, y_test.shape

# Evaluate the model
train_score = rf_model.score(X_train, y_train)
train_rmse = np.sqrt(mean_squared_error(y_train, rf_model.predict(X_train)))
test_score = rf_model.score(X_test, y_test)
test_rmse = np.sqrt(mean_squared_error(y_test, rf_model.predict(X_test)))

## Create dir to the model.
MODEL_DIR = 'model'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print('Directory was created!')

# Save the results of the model.
with open(f'{MODEL_DIR}/results.txt', 'w') as f:
	l1 = 'Laptop Price Prediction'
	l2 = '\n'
	l3 = f'Training score: {train_score}'
	l4 = f'Training RMSE: {train_rmse}'
	l5 = f'Testing score: {test_score}'
	l6 = f'Testing RMSE: {test_rmse}'
	f.writelines([l1, l2, l3, l4, l5, l6])

# Save model.
with open(f'{MODEL_DIR}/model.bin', 'wb') as f:
    pickle.dump((dv, rf_model), f)
    f.close()
    
# Save label encoder objects.
## Save CPU encoder
with open(f'{MODEL_DIR}/cpu_encoder.bin', 'wb') as f:
    pickle.dump((cpu_encoder), f)
    f.close()

## Save GPU encoder
with open(f'{MODEL_DIR}/gpu_encoder.bin', 'wb') as f:
    pickle.dump((gpu_encoder), f)
    f.close()

## Save product encoder
with open(f'{MODEL_DIR}/product_encoder.bin', 'wb') as f:
    pickle.dump((product_encoder), f)
    f.close()

## Save memory encoder
with open(f'{MODEL_DIR}/memory_encoder.bin', 'wb') as f:
    pickle.dump((memory_encoder), f)
    f.close()

## Save typename encoder
with open(f'{MODEL_DIR}/typename_encoder.bin', 'wb') as f:
    pickle.dump((typename_encoder), f)
    f.close()

## Save resolution screen encoder
with open(f'{MODEL_DIR}/resolution_encoder.bin', 'wb') as f:
    pickle.dump((resolution_encoder), f)
    f.close()


