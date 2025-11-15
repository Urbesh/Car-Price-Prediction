# Car-Price-Prediction
This is the first project that I made after learning basics of Machine Learning and Scikit learn

🚀 Project Title & Tagline
==========================

**Car Prediction Project** 🚗
**Predicting Car Prices with Machine Learning** 🤖

📖 Description
===============

The Car Prediction Project is a machine learning-based project that aims to predict car prices based on various features such as make, model, year, mileage, and more. This project uses a combination of data preprocessing, feature engineering, and model selection to achieve high accuracy in predicting car prices. The project is built using Python and utilizes popular libraries such as pandas, numpy, and scikit-learn.

The project consists of two main files: `Car Prediction Project_(old).py` and `Car Prediction Project.py`. The `Car Prediction Project_(old).py` file contains the initial code for loading the data, while the `Car Prediction Project.py` file contains the updated code for loading the data and building the machine learning model. The project uses a pipeline-based approach to build the model, which includes data preprocessing, feature engineering, and model selection.

The goal of this project is to provide a robust and accurate car price prediction model that can be used by car dealerships, buyers, and sellers to estimate the value of a car. The project has the potential to be extended to include additional features such as image recognition, natural language processing, and more. With the rise of online car marketplaces, this project can provide a valuable tool for users to make informed decisions when buying or selling a car.

✨ Features
==========

The Car Prediction Project includes the following features:

1. **Data Preprocessing**: The project includes data preprocessing techniques such as handling missing values, data normalization, and feature scaling.
2. **Feature Engineering**: The project includes feature engineering techniques such as creating new features from existing ones, handling categorical variables, and selecting the most relevant features.
3. **Model Selection**: The project includes model selection techniques such as comparing the performance of different machine learning algorithms and selecting the best one.
4. **Pipeline-Based Approach**: The project uses a pipeline-based approach to build the model, which includes data preprocessing, feature engineering, and model selection.
5. **Stratified Sampling**: The project uses stratified sampling to split the data into training and testing sets, which helps to maintain the same proportion of classes in both sets.
6. **Model Evaluation**: The project includes model evaluation techniques such as mean squared error, mean absolute error, and R-squared to evaluate the performance of the model.
7. **Model Deployment**: The project includes model deployment techniques such as saving the model to a file and loading it for future use.

🧰 Tech Stack Table
====================

| Technology | Description |
| --- | --- |
| **Frontend** | None |
| **Backend** | Python |
| **Tools** | pandas, numpy, scikit-learn, joblib |
| **Libraries** | pandas, numpy, scikit-learn |
| **Frameworks** | None |

📁 Project Structure
=====================

The project structure is as follows:

* `Car Prediction Project_(old).py`: This file contains the initial code for loading the data.
* `Car Prediction Project.py`: This file contains the updated code for loading the data and building the machine learning model.
* `data/`: This folder contains the dataset used for the project.
* `models/`: This folder contains the saved models.
* `utils/`: This folder contains utility functions used in the project.

⚙️ How to Run
==============

To run the project, follow these steps:

1. **Setup**: Install the required libraries by running `pip install -r requirements.txt`.
2. **Environment**: Create a new environment by running `python -m venv env`.
3. **Build**: Build the project by running `python Car Prediction Project.py`.
4. **Deploy**: Deploy the model by saving it to a file using `joblib.dump()`.

🧪 Testing Instructions
=======================

To test the project, follow these steps:

1. **Load the Data**: Load the dataset by running `pd.read_csv('data/data.csv')`.
2. **Split the Data**: Split the data into training and testing sets by running `StratifiedShuffleSplit()`.
3. **Build the Model**: Build the model by running `Pipeline()` and `ColumnTransformer()`.
4. **Evaluate the Model**: Evaluate the model by running `mean_squared_error()` and `mean_absolute_error()`.

