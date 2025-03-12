import numpy as np
import joblib
import pandas as pd
import os

from processing import PreProcessing
from sklearn.preprocessing import label_binarize

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, top_k_accuracy_score, roc_curve, auc, classification_report

import random 
random.seed(4)


class KNNClassifier:
    def __init__(self, X, y, k_range=(1, 15), distance_metric='euclidean', n_splits=10):
        """
        Initializes the KNNClassifier with hyperparameters.
        """
        self.k_range = k_range
        self.distance_metric = distance_metric
        self.n_splits = n_splits
        self.best_k = None
        self.final_model = None
        self.results = {}
        self._X = X
        self._y = y

        
        #X = pd.DataFrame(X, index=df.index)  # Convert back to DataFrame for index manipulation
        #y = y.copy()
        ...

    def fit(self):
        """
        Trains the KNN model using cross-validation to find the best k.
        """
        print(f"Using distance metric: {self.distance_metric}")
    
        auc_scores, f1_scores, accuracy_scores, top_k_scores = [], [], [], []
        best_auc, best_f1, best_accuracy, best_top_k = 0, 0, 0, 0
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for k in range(self.k_range[0], self.k_range[1] + 1):
            knn = KNeighborsClassifier(n_neighbors=k, metric=self.distance_metric)
            
            auc_list, f1_list, acc_list, top_k_list = [], [], [], []
            
            for train_idx, test_idx in kf.split(self._X):
                X_train, X_test = self._X.iloc[train_idx], self._X.iloc[test_idx]
                y_train, y_test = self._y.iloc[train_idx], self._y.iloc[test_idx]
                
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                y_pred_proba = knn.predict_proba(X_test)
                
                auc_list.append(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'))
                f1_list.append(f1_score(y_test, y_pred, average='weighted'))
                acc_list.append(accuracy_score(y_test, y_pred))
                top_k_list = top_k_accuracy_score(y_test, y_pred_proba, k=2)

            # Compute the mean metrics for this k
            avg_auc, avg_f1, avg_accuracy, avg_top_k = np.mean(auc_list), np.mean(f1_list), np.mean(acc_list), np.mean(top_k_list)
            
            auc_scores.append(avg_auc)
            f1_scores.append(avg_f1)
            accuracy_scores.append(avg_accuracy)
            top_k_scores.append(avg_top_k)

            # Track the best k for each metric
            if avg_auc > best_auc:
                best_auc, self.best_k_auc = avg_auc, k
            if avg_f1 > best_f1:
                best_f1, self.best_k_f1 = avg_f1, k
            if avg_accuracy > best_accuracy:
                best_accuracy, self.best_k_accuracy = avg_accuracy, k
            if avg_top_k > best_top_k:
                best_top_k, self.best_top_k = avg_top_k, k

        self.results = {
            'auc_scores': auc_scores,
            'f1_scores': f1_scores,
            'accuracy_scores': accuracy_scores,
            'top_k_scores': top_k_scores,
            'best_k_auc': self.best_k_auc,
            'best_k_f1': self.best_k_f1,
            'best_k_accuracy': self.best_k_accuracy,
            'best_k_top': self.best_top_k
        }

        # Choose the best k based on AUC
        self.best_k = self.best_k_auc

        print(f"\nBest k based on AUC: {self.best_k_auc}")
        print(f"Best k based on F1-Score: {self.best_k_f1}")
        print(f"Best k based on Accuracy: {self.best_k_accuracy}")
        print(f"Best k based on Top-K: {self.best_top_k}")

        self.final_model = KNeighborsClassifier(n_neighbors=self.best_k, metric=self.distance_metric)
        self.final_model.fit(self._X, self._y)

        avg_auc_best_k = auc_scores[self.best_k_auc - self.k_range[0]]
        avg_f1_best_k = f1_scores[self.best_k_f1 -self.k_range[0]]
        avg_accuracy_best_k = accuracy_scores[self.best_k_accuracy - self.k_range[0]]
        avg_top_k_best_k = accuracy_scores[self.best_top_k - self.k_range[0]]

        # Retornando as médias e o melhor modelo treinado
        print(f"\nAverage AUC for best k ({self.best_k}): {avg_auc_best_k}")
        print(f"Average F1-Score for best k ({self.best_k}): {avg_f1_best_k}")
        print(f"Average Accuracy for best k ({self.best_k}): {avg_accuracy_best_k}")
        print(f"Average Top-k Accuracy for best k ({self.best_k}): {avg_top_k_best_k}")

    def predict(self):
        """
        Predicts class labels for the given input data.
        """
        if self.final_model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        return self.final_model.predict(self._X)

    def predict_proba(self):
        """
        Predicts class probabilities for the given input data.
        """
        if self.final_model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        return self.final_model.predict_proba(self._X)

    def get_best_k(self):
        """
        Returns the best k value based on AUC.
        """
        return self.best_k

    def get_results(self):
        """
        Returns the evaluation results from cross-validation.
        """
        return self.results
    
    def save_results_to_csv(self, model_name, filename="models/all_knn_results.csv", clear_file=False):
        """
        Saves the results of a trained model to a CSV file.
        """
        results_df = pd.DataFrame({
            'Model Name': [model_name],
            'k': [self.best_k],
            'Distance Metric': self.distance_metric,
            'AUC': self.results['auc_scores'][self.best_k_auc - self.k_range[0]],
            'F1-Score': self.results['f1_scores'][self.best_k_auc - self.k_range[0]],
            'Accuracy': self.results['accuracy_scores'][self.best_k_auc - self.k_range[0]],
            'Top-K Accuracy': self.results['top_k_scores'][self.best_k_auc - self.k_range[0]]
        })
        if clear_file or not os.path.exists(filename):
            results_df.to_csv(filename, index=False)
        else:
            results_df.to_csv(filename, mode='a', header=False, index=False)

        print(f"Results for {model_name} saved to {filename}")

    def plot_roc_curve(self, y_bin, label, ax):
        y_pred_proba = self.final_model.predict_proba(self._X)

        for i in range(y_bin.shape[1]):  # Loop through each class
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{label} - Class {i} (AUC = {roc_auc:.2f})')

        return self.results

if __name__ == "__main__":
    preprocessor = PreProcessing()
    df_final, df_test = preprocessor.get_data()
    print(df_final.head())
    df_final = preprocessor.replace_outliers(df_final, df_final.columns[3:])
    print(df_final.head())

    X, y = preprocessor.get_feature_target(df_final, train=True)

    # Optionally Oversampling to get better results in imbalanced syndromes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    

    #Training the models
    clear_file = True

    models = {
        "knn_model_cos_resampled": KNNClassifier(X_resampled, y_resampled, (1, 15), "cosine"),
        "knn_model_cos": KNNClassifier(X, y, (1, 15), "cosine"),
        "knn_model_euc_resampled": KNNClassifier(X_resampled, y_resampled, (1, 15), "euclidean"),
        "knn_model_euc": KNNClassifier(X, y, (1, 15), "euclidean")
    }

    for name, model in models.items():
        model.fit()
        model.save_results_to_csv(name, clear_file=clear_file)
        clear_file = False  # Apenas o primeiro modelo limpa o arquivo

    results_file = "models/all_knn_results.csv"
    if not os.path.exists(results_file):
        print("Error: No results found.")
        exit()

    df_results = pd.read_csv(results_file)

    print(f'Models Results:\n {df_results}')

    # choosing the best model
    best_model_row = df_results.loc[df_results['AUC'].idxmax()]
    best_model_name = best_model_row['Model Name']
    best_k = best_model_row['k']
    best_auc = best_model_row['AUC']
    best_distance = best_model_row['Distance Metric']

    print(f"\nBest Model: {best_model_name} | k={best_k} | AUC={best_auc:.4f} | Distance: {best_distance}")

    # Saving the best model
    best_model = models[best_model_name]
    joblib.dump(best_model.final_model, "models/best_knn_model.pkl")
    print(f"\nModel salve in: models/best_knn_model.pkl")



