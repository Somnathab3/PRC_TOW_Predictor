# PRC Data Challenge 2024 - team_honest_turtle

## Introduction

Hello! I’m Somnath Panigrahi, currently pursuing my Master’s degree in Air Transport and Logistics at TU Dresden. In the PRC Data Challenge 2024, I’m competing as part of **team_honest_turtle**. This challenge offers a unique opportunity to apply data science techniques to the aviation sector by accurately estimating the **Actual TakeOff Weight (ATOW)** of flights based on flight and trajectory data.

The PRC Data Challenge aims to develop an open Machine Learning (ML) model capable of estimating ATOW, a critical metric used to calculate fuel consumption and emissions. By moving beyond generic assumptions of ATOW (often set as a fixed percentage of Maximum TakeOff Weight or MTOW), this challenge seeks to create a transparent, reproducible model that supports environmental impact assessments and fuel efficiency improvements.

### Datasets Overview

Two main datasets have been provided for this challenge:

1. **Data for Modeling**: This dataset contains flight details for **369,013 flights** in Europe in 2022, along with high-resolution trajectory data for a majority of these flights. The data has been organized as follows:

   - **Flight List** (`challenge_set.csv`):
     - **Flight Identification**: `flight_id` (unique identifier), `callsign` (obfuscated)
     - **Origin/Destination**: `adep` (departure airport ICAO code), `ades` (destination airport ICAO code), `name_adep`, `name_ades`, `country_code_adep`, `country_code_ades`
     - **Timing**: `date`, `actual_offblock_time`, `arrival_time`
     - **Aircraft Information**: `aircraft_type`, `wtc` (Wake Turbulence Category)
     - **Airline**: `airline` (obfuscated code)
     - **Operational Metrics**: `flight_duration`, `taxiout_time`, `flown_distance`, `tow` (TakeOff Weight in kg)

   - **Trajectory Data**:
     - Each file contains trajectory data at **1-second intervals** in `.parquet` format, resulting in approximately **158 GB** total.
     - **Fields**:
       - `flight_id`, `icao24`: Identifiers for each flight
       - **4D Position**: `longitude`, `latitude`, `altitude`, `timestamp`
       - **Speed and Track**: `groundspeed`, `track`, `track_unwrapped`, `vertical_rate`
       - **Meteorological Data** (where available): `u_component_of_wind`, `v_component_of_wind`, `temperature`

2. **Dataset for Submission**:
   - **Submission Set** (`submission_set.csv`): Contains **105,959 flights** that require ATOW estimates. The structure matches the Flight List, but with an empty `tow` column. For the final ranking, an additional **52,190 flights** will be included in `final_submission_set.csv`.

### Data Access

The data is accessible through two S3 buckets:

- `competition-data/`: Contains data for modeling, including the flight list (`challenge_set.csv`), daily trajectory files, and submission sets.
- `submissions/`: Stores the teams’ submissions for ranking, though contents are not listed publicly per account.

The `competition-data/` bucket is organized as follows:


### Data Types and Importance

Key data types include:

- **Categorical Variables**: `adep`, `ades`, `aircraft_type`, `airline`, `country_code_adep`, `country_code_ades`
- **Numerical Variables**: `flight_duration`, `taxiout_time`, `flown_distance`, `altitude`, `groundspeed`, `vertical_rate`
- **Temporal Variables**: `date`, `actual_offblock_time`, `arrival_time`, `timestamp`

### Real-World Applications

Estimating ATOW accurately has important real-world applications. Precise ATOW estimates allow for:

- **Fuel Efficiency**: By refining fuel load requirements, airlines can reduce fuel waste and minimize costs.
- **Environmental Impact Reduction**: Accurate ATOW is a vital input for emission models that estimate gaseous emissions based on actual fuel burned, supporting aviation’s commitment to environmental sustainability.
- **Operational Planning**: Airlines can optimize routes, plan for better weight distribution, and improve flight safety by understanding the precise takeoff weight requirements.

This challenge underscores the potential for data science to address pressing aviation issues, with an open approach that promotes transparency, reproducibility, and collaboration for a better understanding of aviation's environmental impact.

# Methodology

## Additional Datasets and Data Enrichment

To enhance the predictive accuracy of our model, we integrated several external datasets to provide additional features relevant to aircraft performance, weather, and airport data. Here is an overview of the additional data sources used:

1. **Aircraft Database**:
   - We enriched our flight data with detailed aircraft information from authoritative sources:
     - **EUROCONTROL Aircraft Database**: Provides comprehensive aircraft performance parameters that are critical for assessing fuel burn rates and load capacities.
       - [EUROCONTROL Aircraft Database](https://contentzone.eurocontrol.int/aircraftperformance/default.aspx)
     - **FAA Aircraft Characteristics Database**: Supplies specifications on various aircraft types and their performance profiles.
       - [FAA Aircraft Characteristics Database](https://www.faa.gov/airports/engineering/aircraft_char_database)

2. **Engine Parameters**:
   - By linking aircraft data with **engine performance specifications**, we derived insights into thrust, fuel consumption rates, and altitude-specific performance, which are instrumental in estimating the Actual TakeOff Weight (ATOW).

3. **Weather Data**:
   - Weather data significantly influences fuel burn and load planning. For real-time and historical weather data, we utilized:
     - **Meteostat API**: Offers open-source meteorological data, including wind speed, temperature, and precipitation at various altitudes.
       - [Meteostat API](https://github.com/meteostat)

4. **Airport Data**:
   - **OurAirports Database**: Provided detailed information on airport characteristics, including location, runway dimensions, and operational constraints, which can affect departure and landing efficiency.
     - [OurAirports Database](https://ourairports.com/data/)

5. **Airport Movement Data**:
   - **Eurostat Airport Movement Data (2010-2024)**: Provides historical and current data on airport movements, including aircraft types, flight frequencies, and peak operation periods, which aid in understanding the operational context and potential delays.

6. **ADS-B Data Extraction**:
   - **ADS-B Dataset (285GB)**: To capture high-resolution flight trajectories, we extracted data from a 285GB ADS-B dataset, utilizing specialized libraries:
     - **Traffic**: For handling, filtering, and visualizing ADS-B trajectories.
     - **OpenAP**: Used to analyze flight paths and compute derived features such as rate of climb/descent and in-flight adjustments.

## Model Training and Techniques

For modeling, we experimented with various machine learning approaches, integrating AutoML methods for parameter tuning and leveraging advanced machine learning algorithms to optimize prediction accuracy. Here are the core methods and packages used:

1. **Automated Machine Learning (AutoML)**:
   - We used AutoML to automate hyperparameter tuning and model selection. This allowed for an efficient workflow that explored various modeling techniques without extensive manual configuration.

2. **Machine Learning Algorithms**:
   - **LightGBM**: A gradient boosting framework optimized for speed and performance on large datasets.
   - **CatBoost**: Particularly effective on categorical features, CatBoost helped handle the large volume of categorical data related to airports, aircraft types, and operational patterns.
   - **Neural Networks**:
     - **NNTorch**: A neural network library built on PyTorch, used to design a deep learning model that captures complex non-linear interactions between variables.
     - **Standard Neural Networks**: We also tested custom neural network architectures to explore intricate relationships within the data.

Each of these algorithms was fine-tuned based on feature importance, cross-validation, and hold-out test sets to ensure generalizability of the model.

## Summary of Methodology

By integrating external datasets and employing a robust selection of machine learning techniques, this methodology is designed to produce a reliable and accurate model for estimating ATOW. The combination of high-resolution ADS-B data, enriched aircraft and weather information, and state-of-the-art modeling approaches positions our model to offer substantial real-world applications in fuel efficiency, operational optimization, and environmental impact assessment.

# Feature Description and Sources

## Basic Flight Information

1. **flight_id**: Unique identifier for each flight. *Source: Generated by the data provider.*
2. **date**: Date of the flight. *Source: Provided in the flight list.*
3. **callsign**: Obfuscated callsign for privacy purposes. *Source: Provided in the flight list.*
4. **adep**: Aerodrome of Departure (ICAO code). *Source: Provided in the flight list.*
5. **name_adep**: Name of the departure airport. *Source: OurAirports Database.*
6. **country_code_adep**: Country code for the departure airport. *Source: OurAirports Database.*
7. **ades**: Aerodrome of Destination (ICAO code). *Source: Provided in the flight list.*
8. **name_ades**: Name of the destination airport. *Source: OurAirports Database.*
9. **country_code_ades**: Country code for the destination airport. *Source: OurAirports Database.*
10. **actual_offblock_time**: Actual time the aircraft left the gate. *Source: Provided in the flight list.*
11. **arrival_time**: Arrival time at the destination gate. *Source: Provided in the flight list.*

## Aircraft and Airline Information

12. **aircraft_type**: ICAO aircraft type designator. *Source: EUROCONTROL and FAA Aircraft Database.*
13. **wtc**: Wake Turbulence Category of the aircraft. *Source: EUROCONTROL and FAA Aircraft Database.*
14. **airline**: Obfuscated code for the airline operator. *Source: Provided in the flight list.*

## Flight Timing and Operational Metrics

15. **flight_duration**: Total duration of the flight in minutes. *Source: Calculated from off-block and arrival times.*
16. **taxiout_time**: Duration in minutes from pushback to takeoff. *Source: Calculated from ADS-B data.*
17. **flown_distance**: Great-circle distance (in nautical miles) between departure and arrival airports. *Source: Calculated from geographic coordinates.*
18. **tow**: Estimated TakeOff Weight (TOW) of the flight. *Source: Provided in the flight list.*

## ADS-B Derived Metrics

19. **total_flight_duration_sec_adsb**: Total flight duration from ADS-B data (in seconds). *Source: ADS-B data extraction.*
20. **total_distance_flown_nm_adsb**: Total distance flown based on ADS-B data (in nautical miles). *Source: ADS-B data extraction.*
21. **number_of_position_reports_adsb**: Total number of position reports recorded by ADS-B. *Source: ADS-B data extraction.*

## In-Flight Performance Metrics

22. **average_groundspeed_knots_adsb**: Average groundspeed of the aircraft in knots. *Source: Derived from ADS-B data.*
23. **max_groundspeed_knots_adsb**: Maximum groundspeed recorded during the flight. *Source: Derived from ADS-B data.*
24. **average_vertical_rate_ft_per_min_adsb**: Average rate of climb/descent in feet per minute. *Source: ADS-B data extraction.*
25. **max_vertical_rate_ft_per_min_adsb**: Maximum rate of climb/descent in feet per minute. *Source: ADS-B data extraction.*
26. **num_climb_descent_cycles_adsb**: Number of climb and descent cycles during the flight. *Source: Derived from ADS-B data.*
27. **average_track_change_deg_adsb**: Average change in track direction during the flight (in degrees). *Source: ADS-B data extraction.*

## Meteorological Information (ADS-B Path)

28. **average_temperature_k_adsb**: Average temperature along the flight path (Kelvin). *Source: Meteostat API.*
29. **average_wind_speed_m_per_s_adsb**: Average wind speed along the flight path (meters per second). *Source: Meteostat API.*
30. **average_specific_humidity_adsb**: Average specific humidity along the flight path. *Source: Meteostat API.*
31. **average_altitude_change_rate_ft_per_sec_adsb**: Average rate of altitude change during the flight (feet per second). *Source: ADS-B data extraction.*

## Route Efficiency Metrics

32. **great_circle_distance_nm_adsb**: Great-circle distance calculated from ADS-B trajectory data (nautical miles). *Source: Calculated from ADS-B data.*
33. **route_efficiency_adsb**: Efficiency of the flown route compared to the great-circle distance. *Source: Calculated from ADS-B data.*

## Climb, Cruise, Descent, and Landing Parameters

34-50. **Climb Parameters**: Metrics such as **average speed**, **vertical rate**, **time**, **distance** in the climb phase. *Source: ADS-B and EUROCONTROL/FAA Aircraft Database.*
51-67. **Cruise Parameters**: Includes **cruise speed**, **altitude**, and **duration** in the cruise phase. *Source: ADS-B and EUROCONTROL/FAA Aircraft Database.*
68-84. **Descent Parameters**: Metrics such as **average speed**, **vertical rate**, and **time** during descent. *Source: ADS-B and EUROCONTROL/FAA Aircraft Database.*
85-101. **Landing Parameters**: Includes metrics like **landing speed** and **descent rate** on final approach. *Source: Derived from ADS-B data.*

## Total Fuel Consumption

102. **total_fuel_consumption_kg_adsb**: Total fuel consumed during the flight in kilograms. *Source: Estimated based on aircraft type and flight phase using ADS-B data.*

## Takeoff and Landing Environment Metrics

103-112. **Takeoff Environment**: Average conditions such as **direction**, **temperature**, **vertical rate**, and **altitude** at takeoff. *Source: Meteostat API and ADS-B data.*
113-122. **Landing Environment**: Similar metrics at the landing phase, including **average temperature** and **speed**. *Source: Meteostat API and ADS-B data.*

## Weather Conditions at ADEP and ADES

123-134. **Weather Data for ADEP and ADES**: Includes **temperature**, **dew point**, **relative humidity**, **precipitation**, and **wind speed** at departure and arrival airports. *Source: Meteostat API.*

## Airport Characteristics

135-144. **Airport Runway Details**: Includes **runway length**, **width**, **elevation**, and **latitude/longitude** for both departure and destination airports. *Source: OurAirports Database.*

## Aircraft Performance Specifications (ADB2)

145-174. **Aircraft-Specific Parameters (ADB2)**: Performance specifications such as **MTOW (Maximum TakeOff Weight)**, **initial climb rate**, **cruise speed**, **descent rate**, **approach speed**, **fuel efficiency** metrics, and noise levels. *Source: EUROCONTROL and FAA Aircraft Database.*

## Environmental and Operational Ratios

175-184. **Performance Ratios**: Ratios comparing ADS-B metrics (like speed and climb rate) to ADB2 baseline metrics, assessing performance against standard aircraft specifications. *Source: Derived from ADS-B and ADB2 data.*

## Temperature and Humidity Differences (ADEP - ADES)

185-190. **Environmental Differences**: Differences in **temperature**, **humidity**, and **altitude** between departure and destination. *Source: Meteostat API.*

## Flight Efficiency and Runway Ratios

191-200. **Operational Efficiency Metrics**: Calculated metrics like **flight duration efficiency** and **altitude change rate efficiency**.
201-210. **Runway Ratios**: Ratios between runway lengths and aircraft characteristics (e.g., wing span) at departure and arrival airports. *Source: OurAirports Database and EUROCONTROL/FAA Aircraft Database.*

## Categorical and Derived Features

211-220. **Flight and Route Categories**: Derived categories such as **flight type**, **day or night**, **season**, and **distance category**.
221-230. **Binned Speed Groups**: Includes binned speed groups for takeoff and landing speeds, representing aircraft performance groups.

## Environmental Ratios

231-236. **Barometric and Temperature Ratios**: Comparisons of barometric pressure and temperature at ADEP to aircraft environmental limits. *Source: Derived from Meteostat API and ADB2 data.*


# Flight Take-Off Weight Prediction using Group-Specific Models and Inverse RMSE Weighting

## Introduction

This project aims to predict the **Take-Off Weight (TOW)** of flights using machine learning models. To achieve the best possible **Root Mean Square Error (RMSE)**, we employed multiple grouping methods to train group-specific models. We then applied an **inverse RMSE weighting** strategy to combine the predictions, thereby reducing the overall RMSE impact and improving prediction accuracy.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Approach](#approach)
  - [1. Data Loading and Preprocessing](#1-data-loading-and-preprocessing)
  - [2. Grouping and Merging](#2-grouping-and-merging)
  - [3. Training Group-Specific Models](#3-training-group-specific-models)
  - [4. Training a Global Model](#4-training-a-global-model)
  - [5. Making Predictions](#5-making-predictions)
  - [6. Calculating RMSE for Each Group](#6-calculating-rmse-for-each-group)
  - [7. Inverse RMSE Weighting](#7-inverse-rmse-weighting)
  - [8. Generating Final Predictions](#8-generating-final-predictions)
- [Results](#results)
- [Conclusion](#conclusion)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Acknowledgments](#acknowledgments)

## Dataset

We used two datasets:

- **`challenge_set`**: Training dataset with known TOW values.
- **`submission_set`**: Test dataset for which TOW predictions are required.

Both datasets include various features related to flights, such as aircraft type, flight duration, environmental conditions, and categorical groupings.

## Approach

### 1. Data Loading and Preprocessing

- **Data Loading**: Loaded the datasets and handled any exceptions during the process.
- **Data Optimization**:
  - Downcast numerical columns to reduce memory usage.
  - Converted object columns to categorical types.
- **GPU Availability**: Checked for GPU availability to optimize model training speed using PyTorch.

```python
import pandas as pd
import numpy as np
import logging
import os
import torch

# Define paths and logging
data_dir = r"F:\Project_PRC_Eurocontrol\NEW\2710"
model_dir = os.path.join(data_dir, 'Models')
os.makedirs(model_dir, exist_ok=True)

# Logging setup
log_file_path = os.path.join(model_dir, 'autogluon_training_log.txt')
logging.basicConfig(filename=log_file_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths for datasets
challenge_set_path = os.path.join(data_dir, 'Updated_challenge_set_final_final.csv')
submission_set_path = os.path.join(data_dir, 'Updated_submission_set_final_final.csv')
output_prediction_path = os.path.join(data_dir, 'final_predictions.csv')

# Load datasets
try:
    challenge_set = pd.read_csv(challenge_set_path)
    submission_set = pd.read_csv(submission_set_path)
    logging.info("Datasets loaded successfully.")
except Exception as e:
    logging.error(f"Error loading datasets: {e}")
    raise

# Optimize DataFrames
def optimize_dataframe(df, df_name):
    initial_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info(f"Initial memory usage of '{df_name}': {initial_memory:.2f} MB")

    # Optimize numerical columns
    num_cols = df.select_dtypes(include=['int', 'float']).columns
    df[num_cols] = df[num_cols].apply(pd.to_numeric, downcast='float', errors='ignore')

    # Convert object columns to category
    obj_cols = df.select_dtypes(include=['object']).columns
    df[obj_cols] = df[obj_cols].apply(lambda x: x.astype('category'))

    optimized_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info(f"Optimized memory usage of '{df_name}': {optimized_memory:.2f} MB")
    return df

challenge_set = optimize_dataframe(challenge_set, 'challenge_set')
submission_set = optimize_dataframe(submission_set, 'submission_set')

# Check GPU availability
GPU_AVAILABLE = torch.cuda.is_available()
if not GPU_AVAILABLE:
    logging.warning("GPU is not available. Training will proceed on CPU.")
```

### 2. Grouping and Merging

- **Grouping Columns**: Defined multiple grouping columns to segment the data:
  - `wtc_Flight_distance_category`
  - `wtc_day_or_night_season`
  - `wtc_flight_category`
  - `ADB2_RECAT-EU_Flight_distance_category`
  - `ADB2_RECAT-EU_flight_category`
  - `Toff_speed_binned`
  - `landing_speed_binned`
- **Group Preparation**:
  - Combined the challenge and submission datasets to identify valid groups.
  - Selected groups with at least 100 flights to ensure sufficient data for training.

```python
def prepare_groups(challenge_df, submission_df, group_col, min_flights=100):
    combined_df = pd.concat([challenge_df[[group_col]], submission_df[[group_col]]], ignore_index=True)
    group_counts = combined_df[group_col].value_counts()
    valid_groups = group_counts[group_counts >= min_flights].index.tolist()
    submission_df = submission_df.copy()
    submission_df['group_assigned'] = submission_df[group_col].apply(lambda x: x if x in valid_groups else 'Ungrouped')
    return valid_groups, submission_df

group_columns = [
    'wtc_Flight_distance_category',
    'wtc_day_or_night_season',
    'wtc_flight_category',
    'ADB2_RECAT-EU_Flight_distance_category',
    'ADB2_RECAT-EU_flight_category',
    'Toff_speed_binned',
    'landing_speed_binned',
]

for group_col in group_columns:
    valid_group_names, submission_set = prepare_groups(
        challenge_set,
        submission_set,
        group_col,
        min_flights=100
    )
```

### 3. Training Group-Specific Models

- **Model Training**:
  - For each grouping column, iterated over each valid group.
  - Trained a separate model for each group using AutoGluon's `TabularPredictor`.
  - Saved the trained models for future predictions.
- **Model Storage**:
  - Stored the trained models in a dictionary with keys based on the grouping column and group name.

```python
from autogluon.tabular import TabularPredictor

loaded_group_models = {}

hyperparameters = {
    'GBM': {},
    'CAT': {},
    'XGB': {},
}

for group_col in group_columns:
    for group in valid_group_names:
        try:
            group_data = challenge_set[challenge_set[group_col] == group]
            if group_data.empty:
                continue

            sanitized_group = f"{group_col}_{str(group).replace('/', '_').replace(' ', '_')}"
            model_path = os.path.join(model_dir, f"model_{sanitized_group}")

            predictor = TabularPredictor(label='tow', path=model_path)
            predictor.fit(
                train_data=group_data.drop(columns=['flight_id']),
                hyperparameters=hyperparameters,
                time_limit=200,
                verbosity=1,
                ag_args_fit={}
            )
            loaded_group_models[(group_col, group)] = predictor
        except Exception as e:
            logging.error(f"Failed to train model for group '{group}' in column '{group_col}': {e}")
            continue
```

### 4. Training a Global Model

- **Fallback Model**:
  - Trained a global model on the entire `challenge_set`.
  - Used as a fallback when a group-specific model is not available for a particular group.

```python
global_model_path = os.path.join(model_dir, 'model_global')
global_predictor = TabularPredictor(label='tow', path=global_model_path).fit(
    train_data=challenge_set.drop(columns=['flight_id']),
    hyperparameters=hyperparameters,
    time_limit=100,
    verbosity=2,
    ag_args_fit={}
)
```

### 5. Making Predictions

- **Group-Specific Predictions**:
  - For each flight in the `submission_set`, used the corresponding group-specific model to predict TOW.
  - If a group-specific model was not available, used the global model.
- **Prediction Storage**:
  - Stored all predictions in a DataFrame, with separate columns for each grouping method.

```python
final_predictions = pd.DataFrame()

for group_col in group_columns:
    group_assigned_values = submission_set[group_col].unique()
    for group in group_assigned_values:
        group_rows = submission_set[submission_set[group_col] == group]
        flight_ids = group_rows['flight_id']
        row_features = group_rows.drop(columns=['flight_id'])
        try:
            predictor = loaded_group_models.get((group_col, group), global_predictor)
            predictions = predictor.predict(row_features)
        except Exception as e:
            logging.error(f"Prediction failed for group '{group}' in column '{group_col}': {e}")
            predictions = [np.nan] * len(group_rows)

        group_predictions = pd.DataFrame({
            'flight_id': flight_ids,
            'tow_predicted': predictions
        })

        final_predictions = pd.concat([final_predictions, group_predictions], ignore_index=True)
```

### 6. Calculating RMSE for Each Group

- **RMSE Calculation**:
  - Calculated the RMSE between predicted TOW and actual TOW for each group in the `challenge_set`.
  - Stored the RMSE values for each group and grouping column.
- **Result Storage**:
  - Saved the RMSE results to a CSV file for reference during the weighting process.

```python
from sklearn.metrics import mean_squared_error

rmse_results = []

for group_col in group_columns:
    for group in valid_group_names:
        group_data = challenge_set[challenge_set[group_col] == group]
        actual_tow = group_data['tow']
        sanitized_group = f"{group_col}_{str(group).replace('/', '_').replace(' ', '_')}"
        model_path = os.path.join(model_dir, f"model_{sanitized_group}")

        if os.path.exists(model_path):
            try:
                predictor = TabularPredictor.load(model_path)
                predictions = predictor.predict(group_data.drop(columns=['flight_id', 'tow']))
                rmse_value = mean_squared_error(actual_tow, predictions, squared=False)
                rmse_results.append({
                    'grouping_column': group_col,
                    'group': group,
                    'rmse': rmse_value
                })
            except Exception as e:
                logging.error(f"Failed to make predictions for group '{group}' in column '{group_col}': {e}")
                continue

rmse_results_df = pd.DataFrame(rmse_results)
rmse_results_df.to_csv(output_rmse_path, index=False)
```

### 7. Inverse RMSE Weighting

- **Weight Calculation**:
  - Computed inverse RMSE weights for each group:

    \[
    \text{Weight} = \frac{1}{\text{RMSE} + \epsilon}
    \]

    where \(\epsilon\) is a small constant to prevent division by zero.
- **Weight Mapping**:
  - Mapped the inverse RMSE weights to each flight in the `submission_set` based on their group assignments.

```python
submission_predictions = pd.DataFrame({'flight_id': submission_set['flight_id']})

# Map RMSE values and calculate weights
for grouping_column in group_columns:
    rmse_mapping = rmse_results_df[rmse_results_df['grouping_column'] == grouping_column].set_index('group')['rmse'].to_dict()
    rmse_col_name = f'rmse_{grouping_column}'
    weight_col_name = f'weight_{grouping_column}'
    group_col_name = grouping_column

    # Map RMSE to submission_set
    submission_predictions[rmse_col_name] = submission_set[group_col_name].map(rmse_mapping)
    # Calculate inverse RMSE weight
    submission_predictions[weight_col_name] = 1.0 / (submission_predictions[rmse_col_name] + 1e-6)
```

### 8. Generating Final Predictions

- **Weighted Averaging**:
  - Calculated the weighted average of the predictions from different models for each flight.
  - Gave more influence to models with lower RMSE (higher weights).
- **Handling Missing Values**:
  - Addressed any missing predictions by filling them with the mean of available predictions.
- **Final Output**:
  - Generated the final TOW predictions for each flight.
  - Saved the final predictions to a CSV file.

```python
# Identify predicted_tow columns
predicted_tow_cols = [col for col in submission_predictions.columns if col.startswith('predicted_tow_')]

# Calculate the weighted average of predictions
numerator = pd.Series(0, index=submission_predictions.index, dtype=float)
denominator = pd.Series(0, index=submission_predictions.index, dtype=float)

for grouping_column in group_columns:
    pred_col = f'predicted_tow_{grouping_column}'
    weight_col = f'weight_{grouping_column}'
    if pred_col in submission_predictions.columns and weight_col in submission_predictions.columns:
        weighted_pred = submission_predictions[pred_col] * submission_predictions[weight_col]
        weighted_pred = weighted_pred.fillna(0)
        weight = submission_predictions[weight_col].fillna(0)
        numerator += weighted_pred
        denominator += weight

denominator = denominator.replace(0, np.nan)
submission_predictions['final_tow_prediction'] = numerator / denominator

# Handle missing predictions
submission_predictions['final_tow_prediction'].fillna(submission_predictions[predicted_tow_cols].mean(axis=1), inplace=True)

# Prepare final submission DataFrame
final_predictions_df = submission_predictions[['flight_id', 'final_tow_prediction']].copy()
final_predictions_df.rename(columns={'final_tow_prediction': 'tow'}, inplace=True)

# Save the final predictions to CSV
output_file = data_dir / 'final_tow_predictions.csv'
final_predictions_df.to_csv(output_file, index=False)
```

## Results

By utilizing multiple grouping methods and applying the inverse RMSE weighting strategy, we significantly improved the accuracy of our TOW predictions. Models with lower RMSE had a higher influence on the final prediction, effectively reducing the impact of less accurate models.

The RMSE values for each group were calculated and used to adjust the weights accordingly. This approach allowed us to leverage the strengths of various models trained on different segments of the data.

## Conclusion

The combination of group-specific models and inverse RMSE weighting proved to be an effective strategy for improving the accuracy of flight TOW predictions. By focusing on specific segments of data and adjusting the influence of each model based on its performance, we achieved a lower overall RMSE.

This method can be generalized to other prediction tasks where the data can be meaningfully grouped, and model performance varies across these groups.

## Usage

To reproduce the results or use this approach for your own datasets:

1. **Set Up the Environment**:
   - Install the required dependencies listed below.
   - Ensure you have access to the datasets with the necessary features.

2. **Run the Training Script**:
   - Execute the `Models.py` script to train the group-specific and global models.
   - The script will save the trained models and initial predictions.

3. **Run the RMSE Calculation and Weighting Script**:
   - Execute the `Inverse RMSE.py` script to calculate the RMSE for each group and apply inverse RMSE weighting.
   - The script will generate the final TOW predictions.

4. **Evaluate the Results**:
   - The final predictions will be saved to `final_tow_predictions.csv`.
   - You can analyze the results and compare them with actual TOW values if available.

## Dependencies

- Python 3.6+
- pandas
- numpy
- scikit-learn
- AutoGluon
- PyTorch (for GPU support)
- logging
- pathlib
- re (regular expressions)

Install the required packages using `pip`:

```bash
pip install pandas numpy scikit-learn autogluon torch
```

## Acknowledgments

- **AutoGluon Team**: For providing an excellent open-source AutoML tool.
- **Scikit-Learn Developers**: For the robust machine learning library used for metrics calculation.
- **Community Contributors**: For their valuable insights and code contributions.

---

**Note**: Ensure you have the necessary computational resources, especially if training on large datasets or using complex models. GPU support is recommended for faster training.
