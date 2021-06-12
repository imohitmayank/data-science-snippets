"""Example of a sample code in Pytorch Lightning (on IMDB sentiment dataset)
Author: Mohit Mayank

Dataset link: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
Model: BERT model
"""

# Import
#-----------
# helper
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# for BERT model
from transformers import BertTokenizer, BertModel

# for DL stuff
import torch
from torch.nn import Softmax, Linear
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1

# Load and prepare IMDB dataset
#-------------------------------
# load
df = pd.read_csv("imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")

# divide into test and train
X_train, X_test, y_train, y_test = train_test_split(df['review'].tolist(), df['sentiment'].tolist(), shuffle=True,
                                                    test_size=0.33, random_state=42, stratify=df['sentiment'])

# Define dataset and dataloader
#--------------------------------

# 
def squz(x, dim=0):
    return torch.squeeze(x, dim)
  
# define dataset with load and prep functions. Pass all the data at a time.
class IMDBDataset(Dataset):
    def __init__(self, sentences, labels, max_length=150, model_name='bert-base-uncased'):
        # var
        self.sentences = sentences
        self.labels = [['positive', 'negative'].index(x) for x in labels]
        self.max_length = max_length
        # tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.sentences)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        sentence = self.sentences[index]
        label = self.labels[index]
        # Load data and get label
        X = self.tokenizer(sentence, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        X = {key: squz(value) for key, value in X.items()}
        y = label
        # return
        return X, y
      
# init the train and test dataset
train_dataset = IMDBDataset(X_train, y_train)
test_dataset = IMDBDataset(X_test, y_test)
# create the dataloader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)


# Pytorch lightning model
#--------------------------
class pretrainedBERT(pl.LightningModule):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        # model and layers
        self.BERTModel = BertModel.from_pretrained(model_name)
#         self.BERTModel = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.linear1 = Linear(768, 128)    
        self.linear2 = Linear(128, 2)
        self.softmax = Softmax(dim=1)
        self.relu = torch.nn.ReLU()
        # loss
        self.criterion = torch.nn.CrossEntropyLoss()
        # log
        self.accuracy = Accuracy()
        self.f1 = F1(num_classes=2, average='macro')
    
    def forward(self, x):
        # pass input to BERTmodel
        input_ids, attention_mask = x['input_ids'], x['attention_mask']
        bert_output = self.BERTModel(input_ids, attention_mask=attention_mask)
        output = bert_output.pooler_output     
        output = self.relu(self.linear1(output))
        output = self.linear2(output)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)
        loss = self.criterion(x_hat, y)
        acc = self.accuracy(x_hat.argmax(dim=1), y)
        f1 = self.f1(x_hat.argmax(dim=1), y)
        pbar = {'train_acc': acc, 'train_f1': f1}
        return {'loss': loss, 'progress_bar': pbar}
    
    def validation_step(self, batch, batch_idx):
        loss_dict = self.training_step(batch, batch_idx)
        loss_dict['progress_bar']['val_acc'] = loss_dict['progress_bar']['train_acc']
        loss_dict['progress_bar']['val_f1'] = loss_dict['progress_bar']['train_f1']
        del loss_dict['progress_bar']['train_acc']
        del loss_dict['progress_bar']['train_f1']
        return loss_dict
    
    def training_epoch_end(self, outs):
        avg_train_loss = torch.tensor([x['loss'] for x in outs]).mean()
        avg_train_acc = torch.tensor([x['progress_bar']['train_acc'] for x in outs]).mean()
        avg_train_f1 = torch.tensor([x['progress_bar']['train_f1'] for x in outs]).mean()
        return {'train_loss': avg_train_loss, 'progress_bar': {'avg_train_acc': avg_train_acc, 'avg_train_f1': avg_train_f1}}
        
    def validation_epoch_end(self, outs):
        avg_val_loss = torch.tensor([x['loss'] for x in outs]).mean()
        avg_val_acc = torch.tensor([x['progress_bar']['val_acc'] for x in outs]).mean()
        avg_val_f1 = torch.tensor([x['progress_bar']['val_f1'] for x in outs]).mean()
        return {'val_loss': avg_val_loss, 'progress_bar': {'avg_val_acc': avg_val_acc, 'avg_val_f1': avg_val_f1}}

    def configure_optimizers(self):
        # freezing the params of BERT model
        for name, param in self.named_parameters():
            if 'BERTModel' in name:
                param.requires_grad = False
        # define optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        return optimizer

# Training
#-------------
# init model
model = pretrainedBERT()
# define trainer
trainer = pl.Trainer(gpus=1, max_epochs=5, weights_summary='full')
# fit the trainer
trainer.fit(model, train_dataloader, test_dataloader)

