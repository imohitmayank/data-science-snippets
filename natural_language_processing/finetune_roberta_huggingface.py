"""Finetuning RoBERTa hugging gace (pytorch) model.

Author: Mohit Mayank

- Inspired from Masked Label modelling with BERT article
- Link: https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c
"""

# IMPORT =========
import pandas as pd
from tqdm import tqdm
import numpy as np
import numpy as np

# for deep learning
import torch
import torch.nn as nn
import torch.optim as optim

# load RoBERTa model
from transformers import AdamW
from transformers import RobertaTokenizer, RobertaForMaskedLM

# MODEL LOAD =========
#model_path = "../input/roberta-base" # if local copy is present
model_path = "roberta-base" # if local copy is not present
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForMaskedLM.from_pretrained(model_path)

# DATA PREP 1 =========
data = pd.read_csv("file_with_text.csv")

# tokenize
inputs = tokenizer(data, return_tensors='pt', max_length=250, truncation=True, padding='max_length')
inputs['labels'] = inputs.input_ids.detach().clone()

# create random array of floats with equal dimensions to input_ids tensor
rand = torch.rand(inputs.input_ids.shape)
# create mask array
# mask tokens except special tokens like CLS and SEP
mask_ratio = 0.3
mask_arr = (rand < mask_ratio) * (inputs.input_ids != 101) * \
           (inputs.input_ids != 102) * (inputs.input_ids != 0)

# get the indices where to add mask
selection = []
for i in range(inputs.input_ids.shape[0]):
    selection.append(
        torch.flatten(mask_arr[i].nonzero()).tolist()
    )

# add the mask
for i in range(inputs.input_ids.shape[0]):
    inputs.input_ids[i, selection[i]] = 103
    
# DATA PREP 2 - DATALOADER =========
# define dataset class 
class CommonLitMLMDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)
# create instance
dataset = CommonLitMLMDataset(inputs)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# PRE_TRAIN =============
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# and move our model over to the selected device
model.to(device)
# activate training mode
model.train()

# initialize optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# TRAIN =====================
epochs = 20
for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optimizer.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optimizer.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

# SAVE MODEL =====================
model.save_pretrained("roberta_finetuned_on_commonlit/")
tokenizer.save_pretrained("roberta_finetuned_on_commonlit/")
