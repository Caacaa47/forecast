import json
import joblib
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from data_processor import DataProcessor 

# ==========================================
# Global Model & Processor Loading
# ==========================================
app = FastAPI(title="Forecasting API")

print("--- Loading ML Artifacts ---")
rf_best = joblib.load('best_rf_model.pkl')
processor = joblib.load('data_processor.pkl')

y_idx = processor.scale_cols.index('y')
Y_STD_WATTS = float(processor.scaler.scale_[y_idx])
Y_MEAN_WATTS = float(processor.scaler.mean_[y_idx])
print("--- Models Loaded Successfully! ---")

# ==========================================
# Helper Functions & Endpoints
# ==========================================
def parse_json(json_payload):
    times = json_payload['history']['times']
    data_dict = json_payload['history']['data']
    
    df = pd.DataFrame(data_dict)
    df['times'] = pd.to_datetime(times)
    df.set_index('times', inplace=True)
    df = df.asfreq('15min')
    return df

@app.post("/forecast")
async def generate_forecast(request: Request):
    try:
        payload = await request.json()
        
        labels = payload.get("labels", {})
        parameters = payload.get("parameters", {})
        horizon = parameters.get("horizon", 832)
        
        # Ingest and transform
        raw_df = parse_json(payload)
        clean_df = processor.transform(raw_df)
        
        # Take future horizon rows
        horizon = min(horizon, len(clean_df)) 
        future_df = clean_df.iloc[-horizon:]
        
        X_predict = future_df.drop(columns=['y'])
        future_times = future_df.index
        
        # Predict
        y_pred_scaled = rf_best.predict(X_predict)
        
        # Reverse scale back to Watts (standard Python float conversion)
        y_pred_watts = ((y_pred_scaled * Y_STD_WATTS) + Y_MEAN_WATTS).tolist()
        
        return {
            "labels": labels,
            "parameters": parameters,
            "forecast": {
                "times": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in future_times],
                "data": {
                    "y_forecast_watts": y_pred_watts
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))