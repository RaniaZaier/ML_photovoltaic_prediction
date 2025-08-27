# ML for Photovoltaic Performance Prediction  

This repository contains machine learning workflows to predict **photovoltaic device parameters** (PCE, FF, JSC, VOC) from **molecular structures and descriptors**.  

---

## Overview  

- **Input:** A CSV file (`data-aromatic-param-all.csv`) containing molecular SMILES and photovoltaic parameters.  
- **Processing:**  
  - RDKit is used to generate molecular descriptors from SMILES.  
  - Data is cleaned and combined with experimental/target values.  
- **Machine Learning:**  
  - XGBoost regressors are trained for each target (`PCE`, `FF`, `JSC`, `VOC`).  
  - Model evaluation includes **MSE** and **R²** metrics.  
- **Visualization:**  
  - Correlation heatmap (`heat_map.png`).  
  - True vs Predicted plots for all targets (`train_test.png`).
