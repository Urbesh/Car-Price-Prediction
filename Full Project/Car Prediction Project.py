import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE="Model/model.pkl"
PIPELINE_FILE="Model/pipeline.pkl"

def build_pipeline(num_attribs,cat_attribs):
    # For Numerical Columns 
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    # For Categorical Columns
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),   # fill missing categories if any
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    # Full pipeline
    full_pipeline = ColumnTransformer([
        ("num",num_pipeline, num_attribs),
        ("cat",cat_pipeline, cat_attribs),
    ], remainder="drop")
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # Train the model 
    data=pd.read_csv(f"Dataset/New_car_price_prediction.csv")
    data["Price_cat"]=pd.cut(
        data["Price"],
        bins=[1.0, 8123.0, 18503.0, 26307500.0],
        labels=[1, 2, 3],
        include_lowest=True
    )
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data["Price_cat"]):
        train_set=data.loc[train_index].drop(columns=["Price_cat"]).reset_index(drop=True)
        test_set =data.loc[test_index].drop(columns=["Price_cat"]).reset_index(drop=True)
    test_set["Price_log"]=np.log1p(test_set['Price'])
    test_set.to_csv("Dataset/Price_prediction_testset.csv", index=False)
    
    train_set["Price_log"]=np.log1p(train_set["Price"])
    data_labels=train_set["Price_log"].copy()
    data=train_set.drop(["Price", "Price_log"], axis=1)

    candidate_cat = ["Manufacturer","Model","Category","Leather interior","Fuel type", "Gear box type","Drive wheels","Wheel","Color"]
    cat_attribs = [c for c in candidate_cat if c in data.columns]
    num_attribs = [c for c in data.columns if c not in cat_attribs]

    pipeline=build_pipeline(num_attribs,cat_attribs)
    data_prepared=pipeline.fit_transform(data)

    

    model=RandomForestRegressor(random_state=42)
    model.fit(data_prepared,data_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model Is trained")

    # Final check: show if Price or Price_log are present in pipeline expected inputs
    try:
        expected_cols=[]
        if hasattr(pipeline,"transformers_"):
            for name, trans, cols in pipeline.transformers_:
                if cols is not None:
                    if isinstance(cols, (list, tuple)):
                        expected_cols.extend(list(cols))
                    else:
                        expected_cols.append(cols)
        leakage = [c for c in ("Price", "Price_log") if c in expected_cols]
        if leakage:
            print("WARNING: Pipeline expects these columns (possible leakage):", leakage)
        else:
            print("No Price/Price_log in pipeline feature lists. Good.")
    except Exception:
        pass
else:
    model=joblib.load(MODEL_FILE)
    pipeline=joblib.load(PIPELINE_FILE)
    test_set=pd.read_csv("Dataset/Price_prediction_testset.csv")
    X_test=test_set.drop("Price", axis=1)
    X_test=test_set.drop("Price_log", axis=1)
    y_test=test_set["Price"]
    try:
        # ColumnTransformer stores the original column lists in transformers_ tuples
        num_attribs=pipeline.transformers_[0][2]
        cat_attribs=pipeline.transformers_[1][2]
        # Access fitted SimpleImputer objects from the pipelines
        num_imputer=pipeline.named_transformers_["num"].named_steps["imputer"]
        cat_imputer=pipeline.named_transformers_["cat"].named_steps["imputer"]
        if len(num_attribs) > 0:
            # transform returns a numpy array with imputed values for numeric columns
            test_set[num_attribs]=num_imputer.transform(test_set[num_attribs])
        if len(cat_attribs) > 0:
            test_set[cat_attribs]=cat_imputer.transform(test_set[cat_attribs])
    except Exception:
        pass
    X_test_prepared=pipeline.transform(X_test)
    y_log_pred=model.predict(X_test_prepared)
    predictions=np.expm1(y_log_pred)
    test_set["Predicted_Price"]=predictions
    test_set["Actual_Price"]=y_test
    test_set.drop("Price",axis=1,inplace=True)
    test_set.drop("Price_log",axis=1,inplace=True)
    test_set.to_csv("Output/Predicted_price.csv", index=False)
    print("Inference complete. Saved to Output/Predicted_price.csv")