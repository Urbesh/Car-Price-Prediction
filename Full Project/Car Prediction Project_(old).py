import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# 1. Load the data
data=pd.read_csv("New_car_price_prediction.csv")
# 2. Create a stratified test set based on income category
data["Price_cat"]=pd.cut(
    data["Price"],
    bins=[1.0, 8123.0, 18503.0, 26307500.0],
    labels=[1, 2, 3],
    include_lowest=True
)
split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index,test_index in split.split(data, data["Price_cat"]):
    strat_train_set=data.loc[train_index].reset_index(drop=True).drop("Price_cat", axis=1)
    strat_test_set =data.loc[test_index].reset_index(drop=True).drop("Price_cat", axis=1)
# Work on a copy of training data
data=strat_train_set.copy()
# 3. Separate predictors and labels
data_labels=data["Price"].copy()
data=data.drop("Price",axis=1)
# 4. Define categorical and numeric columns safely
candidate_cat = ["Manufacturer","Model","Category","Leather interior","Fuel type", "Gear box type","Drive wheels","Wheel","Color"]
# only keep categorical columns that actually exist in data
cat_attribs = [c for c in candidate_cat if c in data.columns]
# numeric attributes are everything else
num_attribs = [c for c in data.columns if c not in cat_attribs]
# 5. Pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),   # fill missing categories if any
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
# Full pipeline
full_pipeline = ColumnTransformer([
    ("num",num_pipeline, num_attribs),
    ("cat",cat_pipeline, cat_attribs),
], remainder="drop")
# 6. Transform the data
data_prepared=full_pipeline.fit_transform(data)
# data_prepared is now a NumPy array ready for training
print(data_prepared.shape)