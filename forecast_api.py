import json
import joblib
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from data_processor import DataProcessor 

# ==========================================
# Global Model & Processor Loading
# ==========================================
app = FastAPI(title="Forecasting API")

rf_best = joblib.load('best_rf_model.pkl')
processor = joblib.load('data_processor.pkl')

y_idx = processor.scale_cols.index('y')
Y_STD_WATTS = float(processor.scaler.scale_[y_idx])
Y_MEAN_WATTS = float(processor.scaler.mean_[y_idx])

# Helper Functions & Endpoints

def parse_json(json_payload):
    # 1. Parse historical data
    hist_times = json_payload['history']['times']
    hist_data = json_payload['history']['data']
    df_hist = pd.DataFrame(hist_data)
    df_hist['times'] = pd.to_datetime(hist_times)
    
    # 2. Parse future data (weather forecasts)
    fut_times = json_payload['future']['times']
    fut_data = json_payload['future']['data']
    df_fut = pd.DataFrame(fut_data)
    df_fut['times'] = pd.to_datetime(fut_times)
    
    # Combine history and future safely
    df = pd.concat([df_hist, df_fut], ignore_index=True)

    df.set_index('times', inplace=True)
    df = df.asfreq('15min')
    return df

@app.post("/forecast")
async def generate_forecast(request: Request):
    try:
        payload = await request.json()
        
        if 'future' not in payload:
            raise HTTPException(
                status_code=400,
                detail="'future' block with timestamps and weather data is strictly required to generate a forecast."
            )
               
        labels = payload.get("labels", {})
        parameters = payload.get("parameters", {})
        horizon = parameters.get("horizon", 832)
        
        # 1. Ingest and combine history + future
        raw_df = parse_json(payload)
        
        # 2. Transform everything together to retain interpolation context
        clean_df = processor.transform(raw_df)
        
        # 3. Isolate ONLY the future horizon for prediction
        # (This grabs the final 832 rows of the combined dataset)
        horizon = min(horizon, len(clean_df)) 
        future_df = clean_df.iloc[-horizon:]
        
        # 4. Drop 'y' (which is just NaN or forward-filled dummy data here) to create feature matrix
        X_predict = future_df.drop(columns=['y'])
        future_times = future_df.index
        
        # 5. Predict
        y_pred_scaled = rf_best.predict(X_predict)
        
        # 6. Reverse scale back to Watts (standard Python float conversion)
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