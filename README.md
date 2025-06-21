#  Car Dheko - Used Car Price Prediction

##  Overview

This project aims to build a machine learning pipeline that accurately predicts the **resale price** of used cars based on various **technical specifications** and **categorical attributes**. A fully functional **Streamlit web app** is developed to provide real-time predictions with a user-friendly interface.

##  Dataset Summary

The dataset contains records of used cars scraped from multiple sources and includes:

* **Categorical Features**: `brand`, `model`, `variant`, `fuel_type`, `body_type`, `transmission`, `gear_box`, `drive_type`, `color`, `city`
* **Numerical Features**: `mileage`, `engine_displacement`, `max_power`, `torque`, `top_speed`, `acceleration`, `kms_driven`, `model_year`, `owner_number`
* **Derived Features**: `car_age` (2025 - model\_year)
* **Target Variable**: `price`

##  Data Preprocessing Highlights

* **Missing Value Treatment**: Custom imputation for `mileage`, `max_power`, `torque`, etc.
* **Outlier Removal**: IQR method used on numerical columns
* **Encoding**:

  * Label encoding for `owner_type`
  * One-Hot Encoding for all major categorical fields
* **Feature Engineering**:

  * Derived `car_age` from model year
  * Filtered highly correlated features to avoid multicollinearity

##  Model Building Process

###  Models Compared

| Model               | R2 Score  | MAE (INR)   | RMSE (INR)  | CV R2     |
| ------------------- | --------- | ----------- | ----------- | --------- |
| Linear Regression   | 0.890     | 167,205     | 331,377     | 0.831     |
| Ridge Regression    | 0.891     | 166,745     | 330,668     | 0.832     |
| Lasso Regression    | 0.890     | 167,311     | 331,691     | 0.827     |
| Random Forest       | 0.937     | **111,030** | **250,938** | 0.899     |
| Gradient Boosting   | 0.909     | 161,700     | 301,405     | 0.887     |
| **XGBoost (Final)** | **0.944** | 98,500      | 234,800     | **0.912** |

###  Final Model

* **Selected Model**: XGBoost Regressor
* **Reason**: Highest R² and lowest MAE/RMSE across all evaluations
* **Pipeline Components**:

  * `StandardScaler`
  * `XGBoost Regressor`
  * Saved as `XGBoost_best_car_price_model.pkl`

##  File Structure

```
|— cleaned_car_data.csv               # Cleaned dataset for app use
|— model_features_columns.pkl         # All one-hot encoded features
|— XGBoost_best_car_price_model.pkl   # Final model
|— Streamlitapp.py                    # Deployed Streamlit code
```

##  Streamlit Application Features

* Full interactive form with:

  * Dynamic brand-model-variant filtering
  * Gear box, drive type, transmission controls
  * Sliders and numeric inputs for engine specs
* Shows predicted car price based on inputs
* Visual feedback via `st.success()` and expandable summary

##  Key Takeaways

* Feature importance from XGBoost indicated:

  * `brand`, `engine_displacement`, `kms_driven`, and `fuel_type` are critical drivers
* Minor differences in predictions between body types indicate further domain-specific tuning could help
* Achieved **< 1 lakh MAE** which is acceptable for pricing models in automotive sectors

##  Future Enhancements

* Include **real-time scraped listings** to improve training freshness
* Add **text vectorization** for descriptions or review sentiment
* Deploy with **Docker** or **Cloud** (AWS/GCP) for production use

---

##  Author

**Mugil**
M.Sc. Mathematics | Data Science Fresher | GUVI IITM Certified

> "Driven by data, shaped by logic, delivered with code."
