from processing import PreProcessing
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, top_k_accuracy_score, roc_curve, auc, classification_report

if __name__ == "__main__":
    preprocessor = PreProcessing()
    _, df_test = preprocessor.get_data()
    print(df_test.head())
    df_test = preprocessor.replace_outliers(df_test, df_test.columns[3:])
    print(df_test.head())
    model = joblib.load('models/knn_model.pkl')
    X, y = preprocessor.get_feature_target(df_test)

    print(X)
    print(y)

    model = joblib.load('models/knn_model.pkl')

    y_pred = np.argmax(model.predict_proba(X), axis=1)

    print(y_pred)
    print('----------------------------------------\n Prediciton \n----------------------------------------')
    print('\nAccuracy:', accuracy_score(y, y_pred))
    print('\nClassification Report \n',classification_report(y, y_pred, zero_division=1))