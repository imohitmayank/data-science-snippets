"""Cross validation code
- Algorithm page: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
- I have used Stratified K-Folds cross-validator, you can use any function mentioned above.
"""

# import =======
from sklearn.model_selection import StratifiedKFold

# code =============
# split the dataset into K fold test
def split_dataset(dataset, return_fold=0, n_splits=3, shuffle=True, random_state=1):
    """
    dataset: pandas dataframe
    return_fold: the fold out of `n_split` fold, which is to be returned
    n_splits: # cross fold
    """
    # defined the KFOld function
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    # defined the dataset
    X = dataset
    y = dataset['class'] # label/class

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        if return_fold == i:
            return dataset.loc[train_index], dataset.loc[test_index]

# example call
if __name__ == '__main__':
    # read the dataset
    df = pd.read_csv("....")
    # get one specific fold out of
    train, test = split_dataset(dataset=df, fold=0, n_splits=3)
    # run for all folds
    for fold in range(n_splits):
        train, test = split_dataset(dataset=df, fold=fold, n_splits=n_splits)
        # <perform actions here...>
