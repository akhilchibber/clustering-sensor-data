# Clustering Sensor Data - Use Case 1

<p align="center">
  <img src="https://github.com/akhilchibber/clustering-sensor-data/blob/main/UML_Diagram.png?raw=true" alt="Global Passport Guide">
</p>

## Problem

A machine has 20 sensors recording data during breakdowns. We have 1600 breakdown records. Experts manually labeled only 40 of these into 3 categories which are possibly the reasons for breakdown. The goal is to use those 40 labels to classify the remaining 1560 breakdowns into the same 3 categories.

## Approach

I started by trying unsupervised clustering (K-Means and DBSCAN) to see if the data naturally separates into groups. It didn't as both methods essentially put everything into 1 or 2 clusters, meaning the sensor data doesn't have obvious natural separations. This made me think that I need to use the 40 expert labels as guidance.

I then moved to semi-supervised classification using Label Spreading, which propagates the 40 known labels to the 1560 unknown points based on similarity. The initial accuracy was low (35%), so I improved it through hyperparameter tuning (55%) and SMOTE synthetic data augmentation (77.5%). I also tried PCA to reduce the 20 sensors to fewer features, but found all sensors contribute equally, so PCA didn't help.

To ensure I'm picking the best method, I compared 3 models side by side including (i) Label Spreading, (ii) Random Forest, and (iii) Support Vector Machine. Random Forest achieved 70% accuracy without any synthetic data, while SVM reached 55%. When SMOTE was applied to Random Forest and SVM, they immediately hit 100% accuracy which was a clear sign of overfitting. So SMOTE was only used for Label Spreading.

The final model selected was **Random Forest** because it has the highest honest accuracy (70%) and by honest accuracy we mean accuracy without using synthetic data generated using SMOTE and it also has the highest Confidence Score.

## Key Design Decisions

- **Evaluation method:** Leave-One-Out Cross-Validation on the 40 real labels (hide 1, train on 39, repeat 40 times). This is the most reliable evaluation possible with so few labels.
- **SMOTE only for Label Spreading:** Supervised models (RF, SVM) overfit with SMOTE because synthetic points leak information about the test point. Label Spreading is less affected since it uses the synthetic points as additional anchors in a graph, not as direct training examples.
- **Random Forest as winner:** Despite Label Spreading having higher accuracy (77.5% with SMOTE), Random Forest's 70% is more trustworthy (no synthetic data involved) and its predictions are balanced across all 3 categories of confidence score which are High, Medium, and Low. SVM was rejected because it classified 94% of breakdowns into a single category.

## Assumptions and Tradeoffs

- The 40 expert labels are assumed to be correct and representative of the 3 breakdown categories.
- With only 40 labeled points out of 1600, any model's accuracy is limited. 70% is a strong result given this constraint.
- There is no ground truth for the 1560 predictions, so we rely on confidence scores and category distribution balance as quality indicators.

## What I Would Improve With More Time

- Apply SMOTE inside the cross-validation loop (generate synthetic data only from training fold) to get a fair accuracy estimate for RF and SVM with augmentation.
- Try Self-Training Classifier in which we can iteratively label confident points and retrain.
- Request more expert labels.
- Explore feature engineering e.g. sensor ratios, interactions, etc. to create more discriminative features.
- Build an ensemble that combines predictions from all 3 models using a voting mechanism.

## Folder Structure

```
Take-Home-Assignment/
├── README.md                          ← You are here
├── data_sensors.csv                   ← Input dataset (1600 × 21)
├── Take-Home-Assignment.pdf           ← Original assignment brief
├── Multi_Sensor_Classification.ipynb  ← Main notebook (run this)
├── K_Means.ipynb                      ← Experiment: unsupervised K-Means
├── DBSCAN.ipynb                       ← Experiment: unsupervised DBSCAN
├── PCA_Clustering.ipynb               ← Experiment: PCA investigation
├── models/                            ← Saved trained models (.pkl)
│   ├── random_forest_model.pkl        ← Best model
│   ├── label_spreading_model.pkl
│   ├── svm_model.pkl
│   └── scaler.pkl                     ← Required to scale new data
└── predictions/                       ← Output predictions for 1600 points
    ├── random_forest_predictions.csv  ← Best model predictions
    ├── label_spreading_predictions.csv
    └── svm_predictions.csv
```

## How to Run

Open `Multi_Sensor_Classification.ipynb` in Google Colab and run all cells. It will train all 3 models, evaluate them, and save predictions and model files.
