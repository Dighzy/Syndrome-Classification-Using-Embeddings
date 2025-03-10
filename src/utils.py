import pickle 
import pandas as pd


def load_pickle(path):
    with open (path, 'rb') as f:
        data = pickle.load(f)
        return data
    

def flatten_pickle(data):
    # Flattening the data
    # Iterate over the dictionary
    rows = []
    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                rows.append([str(syndrome_id), str(subject_id), str(image_id)] + list(embedding))

    column_names = ["syndrome_id", "subject_id", "image_id"] + [f"dim_{i}" for i in range(len(rows[0]) - 3)]
    df = pd.DataFrame(rows, columns=column_names)

    print(f'len df: {len(df)}')
    print(df.head(2))
    
    return df

def split_test(df, path_1=None, path_2=None):
    # Split into test and final df
    df_test = df.head(5)
    df_final = df.loc[5:]

    if path_1 == None:
        path_1 ='data/raw/df_train.csv'
    if path_2 == None:
        path_2 = 'data/raw/df_test.csv'

    df_final.to_csv('data/raw/df_train.csv')
    df_test.to_csv('data/raw/df_test.csv')

    print(f'len df: {len(df)}')
    print(f'len df_test: {len(df_test)} save in :')
    print(f'len df_final: {len(df_final)}')

    return df_final, df_test
