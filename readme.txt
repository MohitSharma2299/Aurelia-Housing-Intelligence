# Aurelia — Housing Intelligence

A futuristic, interactive machine-learning web application built with **Streamlit** around a trained **Linear Regression** model for the **California Housing** regression dataset.

The project is designed to be more than a basic ML demo: it combines real model inference, model diagnostics, explainability, scenario analysis, interactive Plotly visualizations, session prediction history, and a custom animated product-style interface.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What the Application Does](#what-the-application-does)
- [Machine Learning Problem](#machine-learning-problem)
- [Dataset](#dataset)
- [Dataset Features](#dataset-features)
- [Target Variable](#target-variable)
- [Model](#model)
- [Training Workflow](#training-workflow)
- [Saved Model Artifact](#saved-model-artifact)
- [Application Architecture](#application-architecture)
- [User Experience and UI](#user-experience-and-ui)
- [Interactive Features](#interactive-features)
- [Visualization System](#visualization-system)
- [Explainability](#explainability)
- [Scenario Analysis](#scenario-analysis)
- [Model Evaluation](#model-evaluation)
- [Prediction History](#prediction-history)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [How Prediction Works](#how-prediction-works)
- [How the Target Is Displayed](#how-the-target-is-displayed)
- [Important Technical Details](#important-technical-details)
- [Limitations](#limitations)
- [Security Notes](#security-notes)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Portfolio / Interview Explanation](#portfolio--interview-explanation)
- [References](#references)

---

# Project Overview

**Aurelia — Housing Intelligence** is an end-to-end machine-learning demonstration application.

It takes a persisted Linear Regression model and turns it into an interactive product where a user can:

1. Enter a California housing profile.
2. Run the trained model.
3. Receive a predicted `MedHouseVal`.
4. See the prediction converted into the dataset's $100,000-unit representation.
5. Explore how individual features contribute to the model output.
6. Change one feature and observe the model's response.
7. Inspect test-set prediction behavior.
8. Inspect residual behavior.
9. Explore fitted model coefficients.
10. Understand the inference flow visually.
11. Review recent predictions made during the current Streamlit session.

The application loads the saved model from `lr_cal_prac.pkl` and loads the California Housing data through scikit-learn. The model artifact and feature-column ordering are used as the source of truth for inference.

---

# What the Application Does

At a high level:

```text
User enters property signals
          ↓
Create a one-row pandas DataFrame
          ↓
Reorder columns to saved training order
          ↓
Persisted Linear Regression model
          ↓
Model prediction
          ↓
Human-readable result
          ↓
Explainability + scenario analysis + diagnostics
```

The product intentionally separates **prediction** from **model understanding**.

It is not only a calculator. It is also an educational and analytical interface for understanding what a Linear Regression model is doing.

---

# Machine Learning Problem

## Problem Type

**Supervised learning — regression**

The model predicts a continuous numeric target rather than a class label.

## Objective

Estimate the median house value for a California census block group from eight numerical features describing:

- income
- house age
- room/bedroom statistics
- population
- occupancy
- latitude
- longitude

## Algorithm

**Linear Regression**

The fitted model can be represented as:

```text
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + β₈x₈
```

Where:

- `ŷ` = predicted `MedHouseVal`
- `β₀` = model intercept
- `βᵢ` = learned coefficient for feature `i`
- `xᵢ` = supplied feature value

The project takes advantage of this structure to make the model relatively easy to inspect and explain.

---

# Dataset

The project uses scikit-learn's:

```python
fetch_california_housing(as_frame=True)
```

The current scikit-learn documentation describes the dataset as:

- **20,640 samples**
- **8 numeric predictive attributes**
- **1 continuous target**
- No missing attribute values
- Target values documented in the range **0.15 to 5**
- Target represented in units of **$100,000**

The dataset was obtained from the **StatLib** repository and is derived from the **1990 U.S. Census**, with one row representing a California census block group.

Official dataset documentation:

https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

Official dataset description:

https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset

Original StatLib source referenced by scikit-learn:

https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

---

# Dataset Features

The model uses exactly these eight features:

| Feature | Meaning | Used by Model |
|---|---|---:|
| `MedInc` | Median income in the block group | Yes |
| `HouseAge` | Median house age in the block group | Yes |
| `AveRooms` | Average number of rooms per household | Yes |
| `AveBedrms` | Average number of bedrooms per household | Yes |
| `Population` | Block-group population | Yes |
| `AveOccup` | Average number of household members / occupancy | Yes |
| `Latitude` | Block-group latitude | Yes |
| `Longitude` | Block-group longitude | Yes |

### Important note about the features

Several columns are not simple raw measurements of one individual house.

For example:

- `AveRooms` is an average at the block-group level.
- `AveBedrms` is an average at the block-group level.
- `Population` describes the block group.
- `AveOccup` is a household-level average.
- Latitude and longitude represent geographic context.

This is therefore a **block-group-level regression problem**, not a modern real-estate listing valuation system for individual houses.

---

# Target Variable

## `MedHouseVal`

The model predicts:

```text
MedHouseVal
```

The scikit-learn California Housing dataset expresses this target in **hundreds of thousands of dollars**.

Example:

```text
2.50
```

corresponds to:

```text
$250,000
```

The Streamlit application converts the raw model result into a dollar-formatted display for readability.

### Very important

The displayed dollar number should **not** be interpreted as a current 2026 market valuation.

The underlying dataset is historical and is derived from the 1990 U.S. Census.

This application demonstrates machine-learning inference and model interpretation; it is not a production real-estate pricing system.

---

# Model

## Algorithm: Linear Regression

The training code supplied for this project follows this basic workflow:

```python
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_squared_error
from sklearn.datasets import fetch_california_housing

dff = fetch_california_housing(as_frame=True).frame

X = dff.drop("MedHouseVal", axis=1)
y = dff["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1234
)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_test_prediction = lr.predict(X_test)

mse = mean_squared_error(y_test, y_test_prediction)
rmse = root_mean_squared_error(y_test, y_test_prediction)

file2 = open("lr_cal_prac.pkl", "wb")

model_bundle = {
    "model": lr,
    "columns": X_test.columns
}

pickle.dump(model_bundle, file2)
file2.close()
```

---

# Training Workflow

The provided training workflow is intentionally simple and transparent.

## 1. Load data

```python
fetch_california_housing(as_frame=True).frame
```

The dataset is loaded as a pandas DataFrame.

## 2. Separate features and target

```python
X = dff.drop("MedHouseVal", axis=1)
y = dff["MedHouseVal"]
```

So:

- `X` contains the eight predictors.
- `y` contains the continuous target.

## 3. Split data

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1234
)
```

This creates:

- **80% training data**
- **20% test data**

The fixed `random_state=1234` makes the split reproducible.

## 4. Train Linear Regression

```python
lr = LinearRegression()
lr.fit(X_train, y_train)
```

The model learns:

- an intercept
- one coefficient for each input feature

## 5. Predict on the test set

```python
y_test_prediction = lr.predict(X_test)
```

## 6. Evaluate

The original workflow calculates:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

The current Streamlit application additionally calculates:

- R²

The application calculates these values at runtime instead of hard-coding them.

---

# What Was NOT Done in the Original Training Workflow

The supplied training code does **not** include:

- feature scaling
- normalization
- standardization
- one-hot encoding
- feature engineering
- polynomial features
- regularization
- hyperparameter search
- cross-validation
- ensemble modeling
- outlier treatment
- missing-value imputation

The application therefore represents the supplied Linear Regression model faithfully rather than implying a more complex training pipeline than actually exists.

---

# Saved Model Artifact

The trained model is stored in:

```text
lr_cal_prac.pkl
```

The project expects the pickle file to contain a dictionary-like object with:

```python
{
    "model": trained_linear_regression_model,
    "columns": feature_columns
}
```

The application loads it with:

```python
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
feature_columns = list(bundle["columns"])
```

The application uses `st.cache_resource` to avoid repeatedly loading the same model object during Streamlit reruns.

---

# Application Architecture

The current application is intentionally implemented as a single Streamlit application file with clearly separated responsibilities.

```text
California_app.py
│
├── Page configuration
├── Custom visual design system
├── Model loading
├── Dataset loading
├── Runtime evaluation
├── Session state
├── Hero / product presentation
├── Model snapshot
├── Prediction laboratory
├── Prediction result
├── Prediction explainability
├── Scenario laboratory
├── Model performance diagnostics
├── Coefficient visualization
├── ML explainer
├── Prediction history
└── Browser-side spotlight interaction
```

---

# User Experience and UI

The interface uses a custom futuristic visual language rather than the default Streamlit appearance.

## Visual direction

The current design uses:

- deep dark background layers
- subtle atmospheric gradients
- glass-like surfaces
- refined borders
- glowing accents
- large editorial typography
- `Manrope` for the primary UI
- `DM Mono` for technical metadata
- soft depth and shadow
- animated hero elements
- interactive hover states

The current page branding is:

```text
Aurelia / Housing Intelligence
```

with the main hero presented as:

```text
Housing
Intelligence
```

---

# Interactive Features

## 1. Live Prediction Laboratory

The user can enter all eight model features:

```text
MedInc
HouseAge
AveRooms
AveBedrms
Population
AveOccup
Latitude
Longitude
```

The inputs are organized into two columns and use contextual tooltips.

The interface then sends the feature values to the saved model.

---

# 2. Human-Friendly Prediction State

Before the first prediction, the result area explains what to do instead of displaying raw HTML or technical debug information.

The empty state communicates:

```text
Your prediction is waiting.
```

and guides the user through:

```text
01 — Describe the property
02 — Run the model
03 — Read the result
```

This is intentionally designed to make the application understandable to a non-technical user.

---

# 3. Prediction Result

After inference, the right-hand panel displays:

- estimated median house value
- raw model value
- model type
- number of signals
- target variable
- live inference status

The displayed dollar value is:

```python
prediction * 100_000
```

because the dataset target is expressed in $100,000 units.

---

# 4. Prediction Explainability

The application uses the mathematical structure of Linear Regression.

For each supplied feature:

```text
contribution = coefficient × feature_value
```

The interface then creates a waterfall-style visualization showing:

```text
Intercept
   +
Feature contribution 1
   +
Feature contribution 2
   +
...
   =
Prediction
```

This makes the internal arithmetic more visible.

### Important interpretation rule

A coefficient or contribution shows the model's fitted relationship.

It does **not** prove that changing a feature will cause the same real-world change.

Therefore the interface explicitly distinguishes model contribution from causation.

---

# 5. Scenario Laboratory

After a prediction has been generated, the user can select one feature and change it using a slider.

The application then:

1. Takes the original input row.
2. Changes one feature.
3. Keeps the other seven features fixed.
4. Runs the actual trained model again.
5. Displays the new raw prediction.
6. Calculates the difference from the original prediction.
7. Displays the change in dollar terms.
8. Plots the response curve.

This creates a practical:

```text
"What happens if this value changes?"
```

experience.

Because the model is linear and the other inputs are held fixed, the response with respect to one feature is expected to follow a linear relationship.

---

# 6. Model Performance Diagnostics

The application generates real diagnostics from the saved model using the same:

```text
test_size = 0.2
random_state = 1234
```

evaluation setup used in the training workflow.

The displayed metrics include:

```text
RMSE
MSE
R²
```

The performance section also includes interactive visualizations for:

### Actual vs Predicted

Each point represents a test-set sample.

The diagonal line represents:

```text
perfect prediction
```

Points farther away from the diagonal represent larger prediction errors.

The points are additionally colored by absolute error.

### Residual vs Predicted

Residual is defined as:

```text
actual - predicted
```

A horizontal zero line is shown.

The chart also displays ±RMSE guide lines when the metric is available.

---

# 7. Interactive Zoom and Pan

The Plotly visualizations are configured for interaction.

Inside the charts:

```text
Mouse wheel      → zoom
Drag             → pan
Double-click     → reset
Hover            → inspect values
```

This is especially helpful for the dense California Housing test-set scatter plots.

---

# 8. Model Coefficient Visualization

The application visualizes the fitted Linear Regression coefficients.

Positive coefficients are shown on one side of the zero reference.

Negative coefficients are shown on the opposite side.

This helps users quickly understand the direction and magnitude of the fitted model relationships.

---

# 9. ML Pipeline Explainer

The application contains a visual explanation of the inference path:

```text
Property Profile
       ↓
Linear Equation
       ↓
MedHouseVal
```

The interface turns this into an animated product-style explanation rather than a plain paragraph.

The three conceptual stages are:

### Signals

The eight numeric inputs describe the property/block-group profile.

### Model

The fitted coefficients and intercept combine those inputs.

### Output

The model produces `MedHouseVal`.

---

# 10. Prediction History

The application maintains a lightweight history in Streamlit session state.

It stores the most recent **8 predictions** made during the current session.

The history includes:

- prediction
- estimated dollar value
- all eight supplied features

This is useful for comparing multiple scenarios without creating a database.

### Important

This history is session-local.

It is not a persistent user database.

Closing/restarting the Streamlit session clears the history.

---

# Technology Stack

## Core Python

- Python
- pandas
- NumPy

## Machine Learning

- scikit-learn
- `LinearRegression`
- `train_test_split`
- `mean_squared_error`
- `root_mean_squared_error`
- `r2_score`

## Web Application

- Streamlit

## Visualization

- Plotly Graph Objects

## Frontend / Motion

- custom HTML
- custom CSS
- JavaScript
- React
- Framer Motion

Framer Motion is loaded inside Streamlit components from a browser-side module URL rather than through a separate npm build.

## Fonts

The application loads:

- Manrope
- DM Mono

through Google Fonts.

---

# Project Structure

Recommended project structure:

```text
Streamlit_UI/
│
├── California_app.py
├── lr_cal_prac.pkl
├── requirements_aurelia.txt
└── README.md
```

## File descriptions

### `California_app.py`

Main Streamlit application.

Contains:

- model loading
- dataset loading
- evaluation
- UI
- prediction logic
- explainability
- scenario analysis
- charts
- session history
- frontend interactions

### `lr_cal_prac.pkl`

Persisted trained model artifact.

Expected to contain:

```python
{
    "model": ...,
    "columns": ...
}
```

### `requirements_aurelia.txt`

Current package list:

```text
streamlit
pandas
numpy
scikit-learn
plotly
```

### `README.md`

Project documentation.

---

# Installation

## 1. Create a virtual environment

From the project directory:

```cmd
python -m venv .venv
```

Activate it on Windows CMD:

```cmd
.venv\Scripts\activate
```

After activation you should see:

```text
(.venv)
```

at the beginning of the terminal line.

---

# 2. Install dependencies

```cmd
python -m pip install --upgrade pip
pip install -r requirements_aurelia.txt
```

For compatibility with the current code, use a recent scikit-learn release because the application imports:

```python
root_mean_squared_error
```

---

# Running the Application

Make sure these two files are in the same directory:

```text
California_app.py
lr_cal_prac.pkl
```

Then run:

```cmd
python -m streamlit run California_app.py
```

Streamlit will provide a local URL, typically similar to:

```text
http://localhost:8501
```

Open that address in the browser.

---

# How Prediction Works

When the user presses:

```text
Run inference ↗
```

the application creates a one-row DataFrame.

Conceptually:

```python
input_df = pd.DataFrame([values])[feature_columns]
```

This ordering is important.

The application does not simply rely on the order in which the controls appear. It explicitly reorders the DataFrame using the saved training columns:

```python
feature_columns
```

Then:

```python
prediction = float(model.predict(input_df)[0])
```

The output is stored in Streamlit session state.

---

# Why the Feature Order Matters

A machine-learning model expects features in the same semantic positions it learned during training.

The project saves:

```python
"columns": X_test.columns
```

inside the pickle artifact so that the inference application can reconstruct the same order.

Without that alignment, the correct numeric values could be passed to the wrong learned coefficients.

This is one of the most important reliability details in the project.

---

# How the Target Is Displayed

The raw model output is a number such as:

```text
2.4607
```

The California Housing target is represented in $100,000 units.

Therefore:

```text
2.4607 × 100,000
=
$246,070
```

The application uses this conversion for the human-readable UI.

It also keeps the raw model value visible so the user can understand the relationship between:

```text
dataset target
```

and:

```text
displayed dollar amount
```

---

# Important Technical Details

## Model loading is cached

The application uses:

```python
@st.cache_resource
```

for model loading.

This avoids unnecessary repeated model deserialization during Streamlit reruns.

## Dataset loading is cached

The California Housing dataset is loaded with:

```python
@st.cache_data
```

This avoids repeatedly fetching/loading the same dataset in the same Streamlit environment.

## Metrics are not hard-coded

RMSE, MSE, and R² are computed from the actual loaded dataset and the actual persisted model.

This prevents the dashboard from displaying fictional performance values.

## Prediction is performed by the persisted model

The Streamlit app does not retrain the model when a user presses the prediction button.

It performs inference using the saved model artifact.

---

# Limitations

This project is intentionally a learning and demonstration application.

## 1. Historical dataset

The California Housing dataset is derived from the 1990 U.S. Census.

It should not be interpreted as a current real-estate pricing dataset.

## 2. Block-group prediction

The rows represent census block groups rather than modern individual property listings.

## 3. Linear model limitations

Linear Regression assumes an additive linear relationship between the supplied features and target.

Real housing markets can contain:

- nonlinear relationships
- interactions
- geographic effects
- market changes
- neighborhood effects
- temporal effects
- complex socioeconomic relationships

A simple linear model cannot capture all of these.

## 4. No uncertainty interval

The current application displays a point prediction.

It does not provide a statistically validated prediction interval or confidence interval.

## 5. Coefficients are not causal explanations

The coefficient/contribution visualizations explain what the fitted equation is doing.

They should not be interpreted as causal estimates.

## 6. Historical market context

The output is not a modern property appraisal and should not be used for financial or real-estate decisions.

---

# Security Notes

## Pickle files

Python pickle files can execute arbitrary Python code during deserialization.

Only load:

```text
lr_cal_prac.pkl
```

or any other pickle artifact from a trusted source.

Do not load unknown `.pkl` files.

## Browser-side resources

The UI loads Framer Motion, React, and fonts through remote browser resources.

This means:

- the core Python application can run locally
- the enhanced motion experience depends on browser access to those resources
- a restricted/offline network may prevent some visual features from loading

---

# Troubleshooting

## Error: `FileNotFoundError` for `/mnt/data/...`

Paths beginning with:

```text
/mnt/data/
```

belong to a different execution environment and should not appear inside your Windows application code.

The local application should use:

```python
Path(__file__).resolve().parent
```

to locate project files.

---

## Error: `streamlit.py` circular import

Do not name your application:

```text
streamlit.py
```

because:

```python
import streamlit
```

may import your local file instead of the installed package.

Use:

```text
California_app.py
```

instead.

---

## Error: `lr_cal_prac.pkl` not found

Make sure:

```text
California_app.py
lr_cal_prac.pkl
```

are in the same directory.

---

## Error: `root_mean_squared_error` is unavailable

Your scikit-learn version may be too old.

Upgrade:

```cmd
pip install --upgrade scikit-learn
```

---

## California Housing data cannot be loaded

The application uses:

```python
fetch_california_housing(as_frame=True)
```

The data may need to be downloaded the first time it is used.

If your environment has restricted network access, the model can still be available from the pickle file, but the application may be unable to populate its runtime diagnostics.

---

# Future Improvements

Possible next iterations include:

## Model improvements

- Ridge Regression
- Lasso Regression
- Elastic Net
- Random Forest Regressor
- Gradient Boosting
- XGBoost
- model comparison
- cross-validation
- hyperparameter tuning
- residual diagnostics by geographic region
- feature scaling experiments

## Explainability improvements

- SHAP for compatible models
- permutation importance
- partial dependence
- confidence/prediction intervals
- local explanation panels

## Data improvements

- current housing data
- time-series analysis
- more granular geospatial data
- external economic indicators

## Product improvements

- persistent prediction database
- user accounts
- saved scenarios
- export to PDF/CSV
- shareable prediction links
- model versioning
- API endpoint
- FastAPI backend
- deployment to a cloud platform
- monitoring and model drift tracking

---

# Portfolio / Interview Explanation

A good way to describe this project in an interview is:

> "I built an interactive ML product around a persisted Linear Regression model trained on the California Housing dataset. The application is implemented in Streamlit with Plotly-based diagnostics and custom frontend components for the product experience. Instead of only exposing a prediction form, I added model explainability, coefficient visualization, residual diagnostics, actual-vs-predicted analysis, a one-feature scenario lab, and session-based prediction history. The application also keeps the model artifact and feature ordering explicit so the inference layer matches the training schema."

---

# Key ML Concepts Demonstrated

This project demonstrates practical understanding of:

- supervised learning
- regression
- train/test splitting
- Linear Regression
- model fitting
- inference
- model persistence
- pickle
- feature ordering
- prediction
- MSE
- RMSE
- R²
- residual analysis
- model coefficients
- local feature contribution
- scenario analysis
- session state
- caching
- interactive visualization
- Streamlit application design

---

# Key Software Engineering Concepts Demonstrated

The application also demonstrates:

- Python project organization
- virtual environments
- dependency management
- reusable data/model loading patterns
- cached resources
- session state
- error handling
- browser-side UI components
- JavaScript integration
- React integration
- Framer Motion integration
- Plotly configuration
- responsive CSS
- interactive product design

---

# References

### Scikit-learn California Housing Loader

https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

### Scikit-learn Real-World Datasets Documentation

https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset

### Original California Housing / StatLib Reference

https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

### Scikit-learn

https://scikit-learn.org/

### Streamlit

https://streamlit.io/

### Plotly

https://plotly.com/python/

---

# Project Status

**Application type:** Interactive ML web application

**Problem type:** Regression

**Model:** Linear Regression

**Dataset:** California Housing

**Inference:** Real persisted model

**Visualization:** Plotly

**UI:** Streamlit + custom HTML/CSS/JavaScript + Framer Motion

**Prediction history:** Session-only

**Target:** `MedHouseVal`

**Model artifact:** `lr_cal_prac.pkl`

---

# Final Note

This project should be viewed as an **educational and portfolio-oriented machine-learning application**.

Its strongest value is not just the prediction itself, but the combination of:

```text
Machine Learning
        +
Model Persistence
        +
Inference
        +
Explainability
        +
Diagnostics
        +
Scenario Analysis
        +
Interactive Visualization
        +
Product UI
```

That combination turns a simple Linear Regression exercise into a complete, demonstrable ML product experience.
