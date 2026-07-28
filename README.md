# Data Scientist Test Case – Electricity Load Forecasting 

Forecasting 15-minute-interval electricity consumption , using
historical load and weather data, with the final model served as a FastAPI endpoint.

## Repository contents

| File | Purpose |
|---|---|
| `solution.ipynb` | Main notebook: EDA, outlier handling, feature engineering, cross-validation, model training/tuning, evaluation, and result plots. |
| `data_processor.py` | `DataProcessor` class — a single fit/transform pipeline (outlier flags, interpolation, scaling, cyclical time features, historical-average feature) shared between training and the API, so training and inference use identical logic. |
| `forecast_api.py` | FastAPI app exposing a `/forecast` endpoint. Loads the trained model + processor and returns predictions for a given horizon. |
| `test_api.py` | Small script that posts the sample `request_body.json` to the running API and sanity-checks the response (correct length, no NaNs). |
| `best_rf_model.pkl` | Trained model (produced by `solution.ipynb`, loaded by the API). |
| `data_processor.pkl` | Fitted `DataProcessor` (produced by `solution.ipynb`, loaded by the API). |

## Approach

**Data.** The JSON contains 15-minute historical load (`y`) plus weather features
(temperature, shortwave radiation, windspeed). Data is parsed, reindexed to a strict
15-minute frequency, and split chronologically into train/test
(last 832 steps held out as the test set, no shuffling to respect time order).

**Cleaning & feature engineering** (`data_processor.py`):
- Values of `y` below a fixed threshold are treated as outlier, set to
  NaN, and linearly interpolated. They are flagged and justified visually in the notebook
  before being applied.
- Weather outliers (extreme radiation, extreme cold/heat) are **not removed**, but
  flagged with binary indicator features, since these are genuine physical extremes
  the model should be aware of, not sensor errors.
- A historical average feature (`y_hist_avg`, grouped by day-of-week × hour) captures
  typical load for that time slot, falling back to a global mean when a slot has too
  few historical observations.
- Cyclical encodings (sin/cos) for hour, day-of-week, and day-of-year, plus a weekend
  flag.
- All fitting (scaler, thresholds, historical averages) is done **only on training
  data** and re-used unchanged on the test/inference data, to avoid leakage.

**Cross-validation.** `TimeSeriesSplit` (3 folds) is used, with the `DataProcessor`
re-fit independently *inside each fold* so no fold ever sees information from its own
validation window. This is what feeds the Bayesian hyperparameter search.

**Models compared:**
- Linear regression (baseline)
- Random Forest (Bayesian-optimized via `skopt.gp_minimize`)
- XGBoost (Bayesian-optimized)
- MLP neural network (Bayesian-optimized over architecture + learning rate + regularization)
- MSTL (multi-seasonal statistical model from `statsforecast`, seasonalities at 96 steps
  = daily and 672 steps = weekly)

Random Forest was selected as the final deployed model based on the accuracy/latency
trade-off (see notebook for full comparison).

## Results

> Full residual/error plots and actual-vs-predicted plots are in `solution.ipynb`.

| Model | Train/tune time (s) | Predict time (s) | MAE (kW) | RMSE (kW) | R² |
|---|---|---|---|---|---|
| Random Forest | 184.52 | 0.033890 | 163.5479 | 220.5080 | 0.9622 |
| XGBoost | 58.47 | 0.004750 | 182.6893 | 241.7121 | 0.9546 |
| MLP | 174.24 | 0.001235 | 194.3585 | 269.0766 | 0.9437 |
| MSTL | 8.40 | 0.041897 | 216.1800 | 285.3657 | 0.9367 |
| Linear Regression | 0.003547 | 0.000279 | 313.3751 | 390.9888 | 0.8811 |

Random Forest gives the best accuracy but is also the most expensive to train (~185s)
and slowest to predict of the ML models; XGBoost is a close second on accuracy at ~3x
faster training and ~7x faster inference, which is a reasonable trade-off if training
time matters more than squeezing out the last bit of RMSE. MSTL is by far the fastest
to train (no hyperparameter search) and gets within ~30% of RF's error, making it a
strong lightweight baseline.

**Exogenous variable / correlation notes:** The correlation matrix shows `y_hist_avg`
is by far the strongest predictor of `y` (r = 0.96) as itcaptures the typical load for 
that day-of-week/hour slot. Of the raw weather variables, shortwave radiation correlates 
more with load (r = 0.47) than temperature (r = 0.25). It's likely because radiation tracks the 
daily solar cycle that also drives occupancy/usage patterns. Temperature's effect on load might 
be more nonlinear (e.g. heating load rises at both very low and very high temperatures). 
Windspeed was tested but only had a weak correlation with `y` (r = 0.13), and including it in the model didn't move accuracy, so it was dropped.

## Running it

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scikit-optimize \
            mlforecast statsforecast joblib fastapi uvicorn requests
```

### 2. Train the model
Run `solution.ipynb` top to bottom. It uses the the sample file `request_body (1).json`
to produce `best_rf_model.pkl` and `data_processor.pkl` in the working directory.

### 3. Start the API
```bash
python -m uvicorn forecast_api:app --host 0.0.0.0 --port 8000
```
(equivalent to `uvicorn forecast_api:app --host 0.0.0.0 --port 8000`; using
`python -m` avoids relying on `uvicorn` being on PATH. `--host 0.0.0.0` binds to all
network interfaces rather than just localhost, and `--port 8000` is what `test_api.py`
expects.)

### 4. Test the API
```bash
python test_api.py
```
Sends the sample file `request_body (1).json` to `POST /forecast` and writes the response to
`forecast_output.json`, checking the forecast length matches the requested horizon and
contains no NaNs.

**Request format:** JSON with a `history` block (`times` + `data`, including weather
features), a `future` block ( timestamps + forecasted weather features for the
horizon being predicted), and `parameters.horizon` (number of 15-minute steps to
forecast). 

**Response format:**
```json
{
  "forecast": {
    "times": ["2024-01-01T00:00:00Z", ...],
    "data": { "y_forecast_watts": [123456.7, ...] }
  }
}
```

## Suggestions for improvement (given more time)

- **Lagged and rolling features with recursive forecasting**: Adding load values from
  previous hours, days, and weeks (plus rolling averages and rolling standard
  deviations).
- **Explicit flags and interaction terms for unusual conditions**: spike-detection
  variables (how far current load deviates from its recent rolling average) and a
  temperature-volatility variable (how far today's peak temperature is from the recent
  average) to help models react to rare but high-impact conditions.
- **Hybrid statistical plus machine learning models**: Combining a statistical baseline
  (from nixtla, e.g. MSTL or ARIMA) that captures the seasonal and trend structure with
  a machine learning model trained on the residuals.
- **Refine features via residual analysis**: plot prediction errors against inputs to
  spot where the model consistently over- or under-predicts. Use these patterns to
  guide targeted feature engineering( e.g., cooling/heating degree-days,
  temperature × hour interactions).

## Time spent & Python background

- Time spent on this assignment: 18 hours
- Prior Python experience: yes