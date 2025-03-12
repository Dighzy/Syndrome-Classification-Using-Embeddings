# Genetic Syndrome Embeddings Analysis

## Project Overview

This project explores the use of K-Nearest Neighbors (KNN) classification to predict syndromes based on embeddings provided in the dataset. The primary focus was on handling challenges such as imbalanced data, outliers, and normalization of embeddings to achieve the best possible performance.

### Key Techniques Used:
- **KNN Classification** for model building.
- **SMOTE (Over-sampling)** to address imbalanced data.
- **Z-score** method to handle outliers.
- **t-SNE** for dimensionality reduction and data visualization.
- **Cross-validation** for hyperparameter tuning.
  
---

## Methodology

1. **Exploratory Data Analysis (EDA):**  
   - Examined the lengths of variables and distributions of the target variable.
   - Identified class imbalance, which led to the implementation of SMOTE to handle the minority class.

2. **Normalization of Embeddings:**  
   - The embeddings were already normalized, but outliers were identified.
   - Used the Z-score method with a threshold of 3 standard deviations to handle outliers.

3. **t-SNE for Data Visualization:**  
   - Applied t-SNE to visually check if the embeddings form natural clusters, confirming that the data is well distributed and suitable for classification.

4. **Augmentation for Imbalanced Data:**  
   - Used SMOTE to over-sample the minority classes and balance the dataset.

5. **Model Building:**  
   - Used the KNN algorithm and performed cross-validation to find the best `k` value.
   - Evaluated performance using AUC, F1-score, Accuracy, and Top-K Accuracy.

---

## Results & Analysis

- The best model was found using `k=15` and **Cosine** distance metric with the resampled dataset.
- The **Cosine metric** outperformed the Euclidean metric in terms of F1-score and other metrics, especially with high-dimensional and non-linear data.
- The performance was significantly improved after addressing class imbalance using SMOTE.

---

## Challenges and Solutions

- **Data Imbalance:**  
  - The dataset had imbalanced classes, which was addressed by using SMOTE to over-sample the minority classes.

- **Outliers:**  
  - Outliers were detected in the embeddings, and the Z-score method was applied to handle them.

- **Normalization:**  
  - The embeddings were normalized, and t-SNE confirmed the natural clustering of the data, making it suitable for KNN classification.

---

## Recommendations for Future Work

- **Augmentation Techniques:**  
  - Add additional augmentation methods such as rotations or transformations to generate more diverse training samples.

- **Hyperparameter Tuning for KNN:**  
  - Further tuning of hyperparameters could improve the model’s performance.

- **Explore More Complex Models:**  
  - While KNN showed good performance, exploring other models like **SVM**, **Random Forest**, or **neural networks** may yield better results.
