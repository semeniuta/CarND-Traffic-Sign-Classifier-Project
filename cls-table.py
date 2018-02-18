import os
import pandas as pd
import numpy as np
import pickle


def classes_counts(y_data):
    unique = np.unique(y_data)
    return [np.sum(y_train == num) for num in unique]


if __name__ == '__main__':

    DATA_DIR = '../traffic-signs-data'
    training_file = os.path.join(DATA_DIR, 'train.p')

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    X_train, y_train = train['features'], train['labels']

    signnames = pd.read_csv('signnames.csv', index_col='ClassId')
    signnames['NumExamples'] = classes_counts(y_train)

    print(signnames)
