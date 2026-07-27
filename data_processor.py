import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.iqr_bounds = {}

        # 1. Explicitly list the exact features you want to KEEP
        self.selected_features = ['shortwave_radiation', 'temperature_2m'] # remove windspeed, less correlated with target
        self.target = ['y']

        # Add the new engineered feature, computed as the historical average of 'y' for the same day of week and hour
        self.engineered_features = ['y_hist_avg']
        
        # Dynamically build the processing lists
        self.scale_cols = self.selected_features + self.target + self.engineered_features
        self.outlier_cols = self.selected_features.copy()
        
        # Define the final output columns 
        self.time_features = ['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 'dayofyear_sin', 'dayofyear_cos'] 
        self.flag_features = ['rad_high_flag','temp_high_flag' ,'temp_low_flag', 'is_weekend']
        self.final_columns = self.scale_cols + self.time_features + self.flag_features

        # Based on visual checks of the hisorical data, set a minimum threshold for 'y' to avoid skewing the historical average with extreme low values
        self.y_min_threshold = 3000000

        # Placeholders for our dynamic thresholds learned during fit()
        self.rad_upper_bound = None
        self.temp_lower_bound = None
        self.temp_high_bound = None

    def _time_features(self, df):
        """Extracts cyclical time features from the datetime index."""

        # minute of hour was also tested but not relevant
        # Keeping both sine and coseine improve the model accuracy
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        
        df['dayofweek_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        df['dayofyear_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)

        # Weeked flag: 1 if Saturday or Sunday, 0 otherwise
        df['is_weekend'] = np.where(df.index.dayofweek >= 5, 1, 0)
        
        return df
    
    def fit(self, df): 
        """Calculates flag thresholds, calculates the historical average, and fits the Z-score scaler for training data"""
        df_fit = df.copy()

        # Learn thresholds for flags instead of capping limits
        q1_rad = df_fit['shortwave_radiation'].quantile(0.25)
        q3_rad = df_fit['shortwave_radiation'].quantile(0.75)
        self.rad_upper_bound = q3_rad + 1.5 * (q3_rad - q1_rad)

        self.temp_lower_bound = df_fit['temperature_2m'].quantile(0.01)
        self.temp_high_bound = df_fit['temperature_2m'].quantile(0.99)

        # Interpolate ONLY 'y' to safely build historical averages
        temp_y = df_fit['y'].copy()
        
        # Apply thresholding before building history so anomalies don't drag down the average
        temp_y.loc[temp_y < self.y_min_threshold] = np.nan 
        temp_y = temp_y.interpolate(method='linear').ffill().bfill()
        
        self.global_y_mean = temp_y.mean()
        
        # Create a temporary DataFrame to hold the day of week and hour for grouping
        # If the count of historical values is less than 5, we will use the global mean to avoid skewing the average with too few samples
        temp_df = pd.DataFrame({'y': temp_y, 'dow': temp_y.index.dayofweek, 'hour': temp_y.index.hour})
        grouped = temp_df.groupby(['dow', 'hour'])['y'].agg(['mean', 'count'])
        grouped['y_hist_avg'] = np.where(grouped['count'] < 5, self.global_y_mean, grouped['mean'])
        
        self.y_avg_map = grouped['y_hist_avg'].to_dict()

        # Create the new feature in df_fit so the Scaler can learn its distribution
        keys = list(zip(df_fit.index.dayofweek, df_fit.index.hour))
        df_fit['y_hist_avg'] = [self.y_avg_map.get(k, self.global_y_mean) for k in keys]

        self.scaler.fit(df_fit[self.scale_cols])
        return self

    def transform(self, df):
        """Applies cleaning, scaling, and feature engineering."""
        df = df.copy()
        
        # Apply feature flags BEFORE any interpolation to capture raw anomalies
        if 'shortwave_radiation' in df.columns:
            df['rad_high_flag'] = np.where(df['shortwave_radiation'] > self.rad_upper_bound, 1, 0)
            
        if 'temperature_2m' in df.columns:
            df['temp_low_flag'] = np.where(df['temperature_2m'] < self.temp_lower_bound, 1, 0)
            df['temp_high_flag'] = np.where(df['temperature_2m'] > self.temp_high_bound, 1, 0)

        # Remove specific target anomalies to NaNs right before interpolation
        if 'y' in df.columns:
            df.loc[df['y'] < self.y_min_threshold, 'y'] = np.nan
            
        # Missing Value Inperpolation
        cols_to_interpolate = self.selected_features + self.target
        for col in cols_to_interpolate:
            if col in df.columns:
                df[col] = df[col].interpolate(method='linear')
                df[col] = df[col].ffill().bfill()

        # Map the historical average feature BEFORE scaling
        keys = list(zip(df.index.dayofweek, df.index.hour))
        df['y_hist_avg'] = [self.y_avg_map.get(k, self.global_y_mean) for k in keys]

        # Apply Standard Scaler 
        df[self.scale_cols] = self.scaler.transform(df[self.scale_cols])
        
        # Extract Cyclical Time Features
        df = self._time_features(df)
        
        # STRICT FEATURE SELECTION
        return df[self.final_columns]