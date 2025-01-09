# **Ukraine Crop Yield Estimation Using Machine Learning**

---

## **Project Overview**

This project focuses on **estimating wheat crop yield in Ukraine** using **MODIS-based variables** and **weather data**, comparing the performance of various machine learning algorithms. The methodology is adapted from the following research paper:

**Ju, S., Lim, H., Ma, J. W., Kim, S., Lee, K., Zhao, S., & Heo, J. (2021). Optimal county-level crop yield prediction using MODIS-based variables and weather data: A comparative study on machine learning models. Agricultural and Forest Meteorology, 307, 108530.**  
[DOI: 10.1016/j.agrformet.2021.108530](https://doi.org/10.1016/j.agrformet.2021.108530)

While the original paper focuses on **county-level crop yield prediction in the United States**, this project targets **Ukraine**, using wheat as the primary crop. The methodology, data preprocessing, and feature selection have been adjusted accordingly.

---

## **Research Objective**

- **Predict wheat yield at a regional level in Ukraine.**
- **Compare the performance of six machine learning algorithms** for yield estimation:
  1. **Support Vector Regression (SVR)**
  2. **Random Forest Regressor (RF)**
  3. **Gradient Boosting Regressor (GB)**
  4. **eXtreme Gradient Boosting (XGB)**
  5. **Decision Tree Regressor (DT)**
  6. **K-Nearest Neighbors Regressor (KNN)**
- Analyze and identify the most suitable machine learning algorithm for yield prediction in Ukraine.

---

## **Data Description**

### **1. MODIS Data**

The following MODIS-based variables were used as features:

- **NDVI**: Normalized Difference Vegetation Index  
- **EVI**: Enhanced Vegetation Index  
- **LST (Day & Night)**: Land Surface Temperature  
- **FPAR**: Fraction of Photosynthetically Active Radiation  
- **LAI**: Leaf Area Index  

These variables were preprocessed using weighted monthly averages, computed over different resolutions (250m, 500m, and 1000m), depending on the variable.

### **2. Weather Data (ERA5 Land)**

The following weather variables were extracted from ERA5 Land data:

- **2m Temperature (t2m)**  
- **Soil Temperature (stl1, stl2)**  
- **Surface Solar Radiation (ssr)**  
- **Total Precipitation (tp)**  
- **Volumetric Soil Water (swvl1, swvl2)**  

---

## **Methodology**

### **Step 1: Data Preprocessing**

1. **MODIS Data Preprocessing**  
   - Monthly weighted averages of MODIS variables were computed.
   - The data was organized into regional-level time series, covering the growing season from March to October (2010–2023).

2. **ERA5 Land Data Preprocessing**  
   - Weather data was extracted and aggregated on a monthly basis for each region.

### **Step 2: Model Training and Evaluation**

- Six different machine learning algorithms were trained on the preprocessed data:
  1. **SVR**  
  2. **Random Forest**  
  3. **Gradient Boosting**  
  4. **XGBoost**  
  5. **Decision Tree**  
  6. **KNN**

- The models were evaluated based on **Mean Absolute Error (MAE)** and **R-squared (R²)** metrics.

---

## **Results**

- The comparison of model performance showed that the **Random Forest Regressor (RF)** and **XGBoost (XGB)** performed the best for wheat yield prediction in Ukraine.
- Detailed results can be found in the [results](results/) folder.

---
