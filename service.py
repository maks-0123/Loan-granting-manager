import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import logging
from datetime import datetime
import json

class ClientData(BaseModel):
    age: int = Field(..., ge = 18, le = 100)
    income: int = Field(..., ge=0)
    gender_cd: Literal["M","F"]
    car_own_flg: Literal["Y","N"]
    education_cd: Literal['SCH', 'UGR', 'GRD']
    appl_rej_cnt: int = Field(..., ge=0)
    Score_bki: float

class PredictionLog: 
    def __init__ (self, log_file = 'predictions.log'):
        self.log_file = log_file
    def log_predictions(self, input_data, prediction, probality):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": input_data,
            "prediction": prediction, 
            "porbality": float(probality),
            'approved':not bool(prediction)
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")

logger = PredictionLog()

app = FastAPI()

model = joblib.load("catboost_model.pkl")
ohe = joblib.load("encoder.pkl")

cat_col = ['gender_cd', 'car_own_flg', 'education_cd']
num_col = ['age', 'appl_rej_cnt', 'income', 'Score_bki']

@app.post("/score")
def score(data: ClientData):
    df = pd.DataFrame([data.dict()])
    cat_encoded = ohe.transform(df[cat_col])
    cat_df = pd.DataFrame(
        cat_encoded, 
        columns = ohe.get_feature_names_out(cat_col))
    final_df = pd.concat(
        [df[num_col].reset_index(drop=True), 
        cat_df.reset_index(drop=True)], 
        axis=1)
    prediction = model.predict(final_df)[0]
    probality = model.predict_proba(final_df)[0][1]

    logger.log_predictions(data, prediction, probality)
    
    approved = not bool(prediction)
    
    print("PROBA FULL:", prediction)
    return {"approved": approved}

