#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

data_1 = pd.read_csv('../input/payco_23.csv')
data_2 = pd.read_csv('../input/payco_2304.csv')
df = pd.concat([data_1,data_2])


# In[2]:


# Step 1: Create user-item matrix
from pandas.api.types import CategoricalDtype
from scipy.sparse import csr_matrix

# Create a new DataFrame with frequency count for each user-item pair
df_grouped = df.groupby(['userid', 'itemid']).size().reset_index(name='frequency')

user_u = list(sorted(df_grouped.userid.unique()))
item_u = list(sorted(df_grouped.itemid.unique()))

user_c = CategoricalDtype(sorted(df_grouped['userid'].unique()), ordered=True)
item_c = CategoricalDtype(sorted(df_grouped['itemid'].unique()), ordered=True)

row = df_grouped['userid'].astype(user_c).cat.codes
col = df_grouped['itemid'].astype(item_c).cat.codes
data = df_grouped['frequency'].tolist()

sparse_matrix = csr_matrix((data, (row, col)), shape=(len(user_u), len(item_u)))

df_user_item = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, index=user_u, columns=item_u)


# In[3]:


# Step 2: Define AutoRec model
import torch
from torch import nn

class AutoRec(nn.Module):
    def __init__(self, num_inputs, hidden_units):
        super(AutoRec, self).__init__()

        self.encoder = nn.Linear(num_inputs, hidden_units)
        self.decoder = nn.Linear(hidden_units, num_inputs)
        
    def forward(self, x):
        x = torch.sigmoid(self.encoder(x))
        x = self.decoder(x)
        return x


# In[4]:


# Step 3: Train and Test AutoRec model
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[5]:


# Hyperparameters
hidden_units = 500
num_epochs = 100
batch_size = 64
learning_rate = 1e-3

model = AutoRec(df_user_item.shape[1], hidden_units).to(device)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=learning_rate)


# In[6]:


# Create DataLoaders
data = torch.FloatTensor(df_user_item.values).to(device)
dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# In[7]:


# Training
model.train()
for epoch in range(num_epochs):
    for i, (inputs,) in enumerate(dataloader):
        inputs = inputs.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")


# In[8]:


# Testing
model.eval()
with torch.no_grad():
    inputs = data
    outputs = model(inputs)
    print(outputs)


# In[18]:


# Step 4: Generate recommendations
import numpy as np

def user_free_inference(items, df_user_item, model, top_k=10):
    # Create a new user vector
    user_vector = np.zeros(df_user_item.shape[1])
    item_indices = []

    # Set the chosen items to the maximum value
    for item in items:
        if item in df_user_item.columns:
            item_index = df_user_item.columns.get_loc(item)
            user_vector[item_index] = df_user_item.values.max()
            item_indices.append(item_index)
        else:
            raise ValueError(f"Item {item} not found in the data")

    # Convert to tensor and move to the correct device
    user_vector = torch.FloatTensor([user_vector]).to(device)

    # Generate recommendations
    with torch.no_grad():
        outputs = model(user_vector)
        predicted_ratings = outputs.cpu().numpy()[0]

    # Remove the chosen items from the predictions
    predicted_ratings[item_indices] = -np.inf

    top_k_item_indices = np.argsort(-predicted_ratings)[:top_k]
    recommended_items = df_user_item.columns[top_k_item_indices]
    recommended_scores = predicted_ratings[top_k_item_indices]

    # Convert item and score to dictionary
    item_score_dict = dict(zip(recommended_items.tolist(), recommended_scores.tolist()))




