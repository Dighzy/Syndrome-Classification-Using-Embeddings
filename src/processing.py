import utils
import pandas as pd
import numpy as np
import pickle
import joblib

class PreProcessing:
    
    def __init__(self, predict=False, data_path='data/raw/mini_gm_public_v0.1.p', threshold_path='models/params/thresholds.pkl'):
        """
        Initializes the preprocessing class.

        :param predict: Set to True for inference, False for training
        :param data_path: Path to the dataset
        :param threshold_path: Path to save/load threshold values
        """
        self.predict = predict
        self.data_path = data_path
        self.threshold_path = threshold_path
        self.thresholds = None 

        if self.predict:
            self.load_thresholds()

    def save_thresholds(self, p1, p99, mean, std):
        """
        Saves the percentile values and statistical values (mean and std) for future use.
        """
        with open(self.threshold_path, 'wb') as f:
            pickle.dump({'p1': p1, 'p99': p99, 'mean': mean, 'std': std}, f)

    def load_thresholds(self):
        """
        Loads the stored threshold values.
        """
        try:
            with open(self.threshold_path, 'rb') as f:
                self.thresholds = pickle.load(f)
        except FileNotFoundError:
            raise ValueError("Threshold file not found! Train the model first.")

    def replace_outliers(self, df, numeric_cols, threshold=3):
        """
        Removes outliers using the z-score method and replaces them with fixed percentiles.
        """

        if not self.predict:
            # Compute mean and standard deviation during training
            mean = df[numeric_cols].mean().astype(np.float32)
            std = df[numeric_cols].std().astype(np.float32)

            p99 = np.percentile(df[numeric_cols], 99, axis=0).astype(np.float32)
            p1 = np.percentile(df[numeric_cols], 1, axis=0).astype(np.float32)

            self.save_thresholds(p1, p99, mean, std)
        else:
            if self.thresholds is None:
                raise ValueError("Thresholds not loaded. Train the model first.")
            p1 = self.thresholds['p1']
            p99 = self.thresholds['p99']
            mean = self.thresholds['mean']
            std = self.thresholds['std']


        z_scores = (df[numeric_cols] - mean) / std

        # Replace outliers with stored percentiles
        df[numeric_cols] = np.where(z_scores > threshold, p99, 
                                    np.where(z_scores < -threshold, p1, df[numeric_cols]))

        return df
    
    @staticmethod
    def get_feature_target(df_final, train=False, mapping_path="models/params/class_mapping.pkl"):
        if train:
            # Convert syndrome_id to categorical codes
            y = df_final['syndrome_id'].astype('category')
            
            class_mapping = dict(enumerate(y.cat.categories))
    
            joblib.dump(class_mapping, mapping_path)

            y = y.cat.codes  

        elif not train:
            # Load the saved mapping
            class_mapping = joblib.load(mapping_path)

            y = df_final['syndrome_id'].map({v: k for k, v in class_mapping.items()})

            y = y.fillna(-1).astype(int)  
        else:
            raise ValueError("Either save_mapping or load_mapping must be True")

        X = df_final.iloc[:, 3:].values
        X = pd.DataFrame(X, index=df_final.index)

        return X, y


    @staticmethod
    def get_data(data_path='data/raw/mini_gm_public_v0.1.p'):
        """
        Loads and processes data from a pickle file.
        """
        data = utils.load_pickle(data_path)
        df = utils.flatten_pickle(data)

        df_final, df_test = utils.split_test(df)

        return df_final, df_test


if __name__ == "__main__":
    preprocessor = PreProcessing()
    df_final, df_test = preprocessor.get_data()
    print(df_final.head())
    df_final = preprocessor.replace_outliers(df_final, df_final.columns[3:])
    print(df_final.head())
    print("Data successfully loaded!")
