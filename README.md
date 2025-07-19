# Smart Farming Yield Prediction 

This project addresses an **unconstrained optimization problem** in machine learning, focusing on a **regression task** to predict **crop yields (kg/ha)** based on data collected from IoT sensors and satellite information. The models are trained on real-world data from 500 smart farms across **India**, **the USA**, and **Africa**.

---

##  Objective
Train regression models to estimate **crop yield** using a combination of:
- Environmental data
- Operational farm data
- Satellite/IoT sensor readings

The goal is to minimize the prediction error using first-order optimization techniques.

---

##  Dataset Overview

The dataset includes the following feature types:

- **Environmental**:  
  `soil_moisture`, `temperature`, `rainfall`, `humidity`, `sunlight_hours`, `soil_pH`

- **Operational**:  
  `irrigation_type`, `fertilizer_type`, `pesticide_usage`, `growth_duration`

- **Geolocation & Satellite**:  
  `latitude`, `longitude`, `NDVI_vegetation_index`

- **Categorical**:  
  `region`, `crop_type`, `crop_disease_status`

- **Target Variable**:  
  `yield_kg_per_hectare`

- **Excluded Features**:  
  `farm_id`, `sensor_id`, `sowing_date`, `harvest_date`, `timestamp`

---

##  Preprocessing Steps

- Label encoding of categorical variables
- Feature normalization using `StandardScaler`
- Random train/test split for model evaluation

---

##  Optimization Methods Implemented

We implemented and compared two first-order optimization algorithms to solve the regression problem:

- `Gradient Descent` – classic batch-based method
- `Stochastic Gradient Descent` – online learning with randomized batches

Implemented in the following files:

- `gradient_descent.m`
- `stochastic_gradient.m`

---

##  Files

- `Smart_Farming_Crop_Yield_2024.csv` — primary dataset
- `main.m` — main script for running training and evaluation
- `gradient_descent.m`, `stochastic_gradient.m` — optimization algorithms
- `split_data.m` — helper function to partition the dataset
- `cosid.m`, `cosid_deriv.m` — custom loss/cost and gradient functions
- `baza.txt`, `.DS_Store`, `321AA_Palade_Madalina_Ioana.pdf` — misc and documentation

---

##  Future Improvements

- Add regularization (L1/L2) to improve generalization
- Explore non-linear models (e.g., neural networks or decision trees)
- Hyperparameter tuning with grid/random search
- Integration with visualization dashboards (e.g., for yield maps)

---

##  Author

**Mădălina Ioana Palade**  
Faculty of Automatic Control and Computers  
Bucharest, Romania

---

## 📄 License

This project is academic in nature. For inquiries about data usage or results, please contact the author.

