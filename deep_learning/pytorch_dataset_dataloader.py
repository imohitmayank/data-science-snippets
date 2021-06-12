"""How to define handle datasets in Pytorch
Author: Mohit Mayank
Dataset: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
"""

# Import
# -----------
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Load and prepare IMDB dataset
#-------------------------------
# load
df = pd.read_csv("imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")

# divide into test and train
X_train, X_test, y_train, y_test = train_test_split(df['review'].tolist(), df['sentiment'].tolist(), shuffle=True,
                                                    test_size=0.33, random_state=42, stratify=df['sentiment'])

# Dataset
# -----------
# define dataset class which takes care of the dataset preparation before passing to model.
# Class takes all data at once (__init__) and define functions to fetch one data at a time (__getitem__)
class IMDBDataset(Dataset):
    def __init__(self, sentences, labels, max_length=150):
        'constructor'
        # var
        self.sentences = sentences
        self.labels = [['positive', 'negative'].index(x) for x in labels]
        self.max_length = max_length
        # tokenizer
        self.tokenizer = ... # some tokenizer

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.sentences)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        sentence = self.sentences[index]
        label = self.labels[index]
        # Load data and get label
        X = self.tokenizer(sentence, ...) # tokenize one data sample
        y = label
        # return
        return X, y

# Init and Dataloader
# ---------------------
# init the train and test dataset
train_dataset = IMDBDataset(X_train, y_train)
test_dataset = IMDBDataset(X_test, y_test)
# create the dataloader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
