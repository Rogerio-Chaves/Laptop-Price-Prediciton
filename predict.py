# Import libraries.
from flask import Flask
from flask import request, jsonify
from sklearn.feature_extraction import DictVectorizer

import pandas as pd
import pickle
import re


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
        

# Load the model.
with open('model/model.bin', 'rb') as f:
	model = pickle.load(f)
	f.close()

# Load relevant columns for the model.
with open('model/relevant_columns.bin', 'rb') as f:
	relevant_columns = pickle.load(f)
	f.close()   

relevant_columns.remove('price_euros') 
    
# Load the label encoder files.
with open('model/cpu_encoder.bin', 'rb') as f:
	cpu_encoder = pickle.load(f)
	f.close() 
with open('model/gpu_encoder.bin', 'rb') as f:
	gpu_encoder = pickle.load(f)
	f.close() 
with open('model/product_encoder.bin', 'rb') as f:	
	product_encoder = pickle.load(f)
	f.close() 
with open('model/memory_encoder.bin', 'rb') as f:	
	memory_encoder = pickle.load(f)
	f.close()
with open('model/typename_encoder.bin', 'rb') as f:	
	typename_encoder = pickle.load(f)
	f.close() 
with open('model/resolution_encoder.bin', 'rb') as f:	
	resolution_encoder = pickle.load(f)
	f.close() 
    
 
app = Flask('laptop_price_prediction')

@app.route('/predict', methods=['POST'])
def predict():
	sample = request.get_json()
	sample = pd.Series(sample)
	
	X = sample[relevant_columns]
    
	# hange string values to numeric values
	X["ram"] = int(to_numeric(X["ram"]))
	X["weight"] = float(to_numeric(X["weight"]))

	# Apply label encoding on training dataset
	X['cpu'] = cpu_encoder.transform([X['cpu']])[0]
	X['gpu'] = gpu_encoder.transform([X['gpu']])[0]
	X['product'] = product_encoder.transform([X['product']])[0]
	X['memory'] = memory_encoder.transform([X['memory']])[0]
	X['typename'] = typename_encoder.transform([X['typename']])[0]
	X['screenresolution'] = resolution_encoder.transform([X['screenresolution']])[0]
	
	X = X.values
	
	y_pred = model.predict(X.reshape(1, -1))
       
	result = {
        	'price': float(y_pred),
	}
    
	return jsonify(result)
	
if __name__ == '__main__':
	 app.run(debug=True, host='0.0.0.0', port=9696)

