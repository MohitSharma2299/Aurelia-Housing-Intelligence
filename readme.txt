# ✦ Aurelia — Housing Intelligence

> **An interactive Linear Regression experience for California housing data.**

A futuristic Streamlit app that turns a trained ML model into an interactive product — not just a prediction form.

---

## ⚡ What it does

**Enter property signals → Run the model → Understand the prediction**

- 🏠 Predict `MedHouseVal`
- 🔍 Explore feature contributions
- 🎛️ Run **What-If** scenarios
- 📊 Inspect model performance
- 🧠 Understand Linear Regression visually
- 🕘 Compare recent predictions in the current session

---

## 🧠 The ML Model

| | Details |
|---|---|
| **Dataset** | California Housing |
| **Model** | Linear Regression |
| **Target** | `MedHouseVal` |
| **Features** | 8 |
| **Test split** | 80 / 20 |
| **Random state** | `1234` |

**Features:** `MedInc` · `HouseAge` · `AveRooms` · `AveBedrms` · `Population` · `AveOccup` · `Latitude` · `Longitude`

Model artifact:

```text
lr_cal_prac.pkl
```

The app loads the persisted model and preserves the saved feature order during inference.

---

## 📊 Dataset

The project uses scikit-learn's **California Housing** dataset.

**20,640 samples · 8 predictors · continuous target**

`MedHouseVal` is represented in **$100,000 units**.

```text
2.46 → $246,000
```

> This is a historical ML dataset, not a current real-estate valuation system.

---

## ✨ Product Experience

### Prediction Laboratory
Enter the eight model inputs and run real inference.

### Explainability
Explore coefficients and feature-level contributions.

### Scenario Lab
Change one feature and see how the actual model responds.

### Model Diagnostics
Interactive:

- Actual vs Predicted
- Residual analysis
- RMSE / MSE / R²
- Coefficient visualization

### Futuristic UI
Built with:

- Custom CSS
- Framer Motion
- React
- Plotly
- Streamlit

Charts support **hover, zoom, pan and reset**.

---

## 🏗️ Stack

```text
Python
├── pandas / NumPy
├── scikit-learn
├── Streamlit
├── Plotly
└── React + Framer Motion
```

---

## 📁 Project Structure

```text
Streamlit_UI/
├── California_app.py
├── lr_cal_prac.pkl
├── requirements_aurelia.txt
└── README.md
```

---

## 🚀 Run Locally

### 1. Create environment

```bash
python -m venv .venv
```

### 2. Activate on Windows

```cmd
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements_aurelia.txt
```

### 4. Start

```bash
python -m streamlit run California_app.py
```

Open:

```text
http://localhost:8501
```

---

## 🔬 How Prediction Works

```text
8 Features
    ↓
Saved Linear Regression Model
    ↓
Raw MedHouseVal
    ↓
Human-readable $ value
```

The application **does not retrain the model** during prediction.

---

## ⚠️ Limitations

- Historical dataset
- Not a modern property appraisal
- Linear model captures only linear relationships
- Feature contributions are **not causal claims**
- Prediction history is session-only

---

## 🎯 Why This Project?

This project combines:

**Machine Learning + Explainability + Data Visualization + Product Design**

Instead of stopping at:

```python
model.predict()
```

it turns the model into something users can **explore, understand and interact with**.

---

## 📚 References

- [Scikit-learn — California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/python/)
- [Framer Motion](https://motion.dev/)

---

### Built for learning. Designed like a product.
