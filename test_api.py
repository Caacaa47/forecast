import json
import requests
import math

# 1. Load the original request JSON you want to send
with open("request_body (1).json", "r") as f:
    payload = json.load(f)

print("Sending data to the API...")

# 2. Send the POST request to your running FastAPI server with a timeout and error handling
try:
    response = requests.post("http://127.0.0.1:8000/forecast", json=payload, timeout=30)
except requests.exceptions.ConnectionError:
    print("Couldn't connect to the API. Did you start it with uvicorn?")
    exit(1)

# 3. Output and validate the result
if response.status_code == 200:
    forecast_data = response.json()
    
    times = forecast_data["forecast"]["times"]
    values = forecast_data["forecast"]["data"]["y_forecast_watts"]
    expected_horizon = payload.get("parameters", {}).get("horizon", 832)

    # Check 1: did we get the number of points we asked for?
    assert len(times) == len(values) == min(expected_horizon, len(times)), \
        "Mismatch between horizon and returned forecast length!"

    # Check 2: are any of the predicted values NaN (i.e. broken)?
    nan_count = sum(1 for v in values if v is None or math.isnan(v))
    assert nan_count == 0, f"Found {nan_count} NaN values in the forecast!"

    with open("forecast_output.json", "w") as f:
        json.dump(forecast_data, f, indent=2)
    print(f"Success! {len(values)} forecast points saved, no NaNs found.")
else:
    print(f"Error {response.status_code}: {response.text}")