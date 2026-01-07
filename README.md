# 🚑 AI-Driven Emergency Control System

## 📌 Project Overview
This project is an intelligent emergency management system designed to help city authorities and emergency responders select the most efficient hospital routes during critical situations. It combines **machine learning, graph algorithms, and simulated traffic data** to identify the shortest and least congested paths for emergency vehicles, ensuring faster response times and better decision-making.

---

## 🎯 Key Features
- **Predicts traffic congestion** using Random Forest and XGBoost models on simulated traffic data.
- **Optimizes routes** for emergencies considering distance and traffic conditions.
- **Interactive visualization** of city traffic and emergency paths using Streamlit.
- **Emergency notifications** to on-duty traffic officers.
- **Simulated traffic data** for vehicle count, speed, and congestion for 24 hours.

---

## 🧠 Machine Learning & Analytics
- Trains **Random Forest** and **XGBoost regressors** on simulated traffic datasets to predict congestion scores.
- Evaluates models using **R² Score, MAE, and RMSE**.
- Clusters roads into traffic intensity levels using **K-Means**.

---

## 🗺 Route Optimization
- **Shortest Path Algorithm** finds the minimum distance route.
- **Least Traffic Path Algorithm** dynamically adjusts routes based on predicted congestion.

---

## 📊 Visualizations
- City traffic map with nodes (areas) and edges (roads).
- Highlights shortest path in **light blue** and least congested path in **pink dashed** lines.
- Predicted vs actual congestion plots.
- Dashboard comparing model performance.

---

## 🛠 Technologies Used
- **Python:** Pandas, NumPy, Matplotlib, NetworkX  
- **Machine Learning:** XGBoost, Random Forest, K-Means  
- **Visualization & Web App:** Streamlit  
- **Other Tools:** Jupyter Notebook  

---

## 🚀 Future Enhancements
- Integrate with **real-time traffic APIs** for live data.
- **GPS-based ambulance tracking**.
- Automated **SMS/email notifications** to officers.
- Deployment on **cloud platforms** for real-time usage.
