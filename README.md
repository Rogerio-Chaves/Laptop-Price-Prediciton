# Laptop Price Prediciton

![Laptop Price Prediciton](/imgs/laptop-homepage.jpg "Laptop Price Prediciton")

## Description:

This project aim to predict the laptop price. For this it use the [Laptop Price Dataset](https://www.kaggle.com/datasets/muhammetvarl/laptop-price) available on Kaggle. The dataset contains 1303 instaces with 12 columns (11 features and 1 target).

**Columns:**

`Company:` Laptop Manufacturer</br>
`Product:` Brand and Model</br>
`TypeName:` Type (Notebook, Ultrabook, Gaming, etc.)</br>
`Inches:` Screen Size</br>
`ScreenResolution:` Screen Resolution</br>
`Cpu:` Central Processing Unit (CPU)</br>
`Ram:` Memory RAM</br>
`Memory:` Hard Disk / SSD Memory</br>
`GPU:` Graphics Processing Units (GPU)</br>
`OpSys:` Operating System</br>
`Weight:` Laptop Weight</br>
`Price_euros:` Price (Euro)

**Dataset author:** Muhammet Varli.

## Project Structure:

	├── data <- Folder containing the dataset used in the project.
		└── laptop_price.csv <- full dataset.
		│
		└── test.csv <- Test dataset used in the predict step.
	│ 
	├── imgs <- Folder containing images used in the README file.
	│ 
	├── model <- Folder containing the model files.
		│
		└── cpu_encoder.bin <- CPU label encoder.
		│
		└── gpu_encoder.bin <- GPU label encoder.
		│
		└── memory_encoder.bin <- Memory label encoder.
		│
		└── model.bin <- The trained model.
		│
		└── product_encoder.bin <- Product label encoder.
		│
		└── relevant_columns.bin <- Relevant columns generates of the feature selection step.
		│
		└── resolution_encoder.bin <- Resolution label encoder.
		│
		└── results.txt <- Results of the model training.
		│
		└── typename_encoder.bin <- Typename label encoder.
	│ 
	├── Pipfile <- File for the virtual environment.
	│ 
	├── Pipfile.lock <- File for the virtual environment.
	│ 
	├── README.md
	│ 
	├── notebook.ipynb <- This notebook contains the preprocessing, EDA and model selection.
	│ 
	├── requirements <- This file contains the library dependencies of the project.
	│ 
	├── train.py <- Python script for the training step.


## How to run this project:

### Get the datasets

There are two ways to get the dataset:

- Original is available on the Kaggle in this <a href="https://www.kaggle.com/datasets/muhammetvarl/laptop-price">link</a>.
- Data folder in this repository.

### Activate the virtual environment


## Colaborators:

`Rogerio Chaves`
