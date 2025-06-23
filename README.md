***Smart Farming Yield Prediction***

This project addresses an *unconstrained optimization problem* in *deep learning*, focusing on a *regression task* to predict crop yields (kg/ha) using *machine learning*.

**Objective**
Train regression models to estimate crop yield based on environmental, operational, and satellite/IOT-derived data from 500 smart farms across India, the USA, and Africa.

**Dataset overview**
Features include:

- Environmental: soil moisture, temperature, rainfall, humidity, sunlight, pH
- Operational: irrigation type, fertilizer type, pesticide usage, growth duration
- Geolocation: latitude, longitude, NDVI vegetation index
- Categorical data: region, crop type, crop disease status

   *Target*: yield_kg_per_hectare

Excluded: farm_id, sensor_id, sowing_date, harvest_date, timestamp

**Preprocessing**
- Label encoding for categorical variables
- Feature normalization (StandardScaler)
- Train/test split

**Optimization methods**
- Implemented first-order optimization algorithms:
- Gradient Descent
- Stochastic Gradient Descent 

