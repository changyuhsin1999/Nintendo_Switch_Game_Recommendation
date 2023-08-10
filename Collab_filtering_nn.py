import urllib
import zipfile
import time

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity

def nn_dataset():
    #Load and clean dataset
    raw_df = pd.read_csv('/content/Nintendo_Switch_Game_Recommendation/switch-games.csv')
    game_rating_df = raw_df.loc[:,['id','title','rating']].dropna()
    game_rating_df = game_rating_df[game_rating_df.rating != 0.0]
    game_rating_df['id_num'] = pd.Series(game_rating_df['id']).astype('category')
    game_rating_df['game_num'] = pd.Series(game_rating_df['title']).astype('category')
    game_rating_df['id_num'] = game_rating_df['id_num'].cat.codes
    game_rating_df['game_num'] = game_rating_df['game_num'].cat.codes
    return game_rating_df


df = nn_dataset()
X = df.loc[:,['id_num','game_num']]
y = df.loc[:,'rating']

# Split our data into training and test sets
X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=0, test_size=0.2)

def dataloaders(X_train,y_train,X_val,y_val,batch_size):
    # Convert training and test data to TensorDatasets
    trainset = TensorDataset(torch.from_numpy(np.array(X_train)).long(),
                            torch.from_numpy(np.array(y_train)).float())
    valset = TensorDataset(torch.from_numpy(np.array(X_val)).long(),
                            torch.from_numpy(np.array(y_val)).float())

    # Create Dataloaders for our training and test data to allow us to iterate over minibatches
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader

batchsize = 16
trainloader,valloader = dataloaders(X_train,y_train,X_val,y_val,batchsize)

class NNColabFiltering(nn.Module):

    def __init__(self, n_users, n_items, embedding_dim_users, embedding_dim_items, n_activations, rating_range):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_embeddings=n_users,embedding_dim=embedding_dim_users)
        self.item_embeddings = nn.Embedding(num_embeddings=n_items,embedding_dim=embedding_dim_items)
        self.fc1 = nn.Linear(embedding_dim_users+embedding_dim_items,n_activations)
        self.fc2 = nn.Linear(n_activations,1)
        self.rating_range = rating_range

    def forward(self, X):
        # Get embeddings for minibatch
        embedded_users = self.user_embeddings(X[:,0])
        embedded_items = self.item_embeddings(X[:,1])
        # Concatenate user and item embeddings
        embeddings = torch.cat([embedded_users,embedded_items],dim=1)
        # Pass embeddings through network
        preds = self.fc1(embeddings)
        preds = F.relu(preds)
        preds = self.fc2(preds)
        # Scale predicted ratings to target-range [low,high]
        preds = torch.sigmoid(preds) * (self.rating_range[1]-self.rating_range[0]) + self.rating_range[0]
        return preds
    

def train_model(model, criterion, optimizer, dataloaders, device, num_epochs=5, scheduler=None):
    model = model.to(device) # Send model to GPU if available
    since = time.time()

    costpaths = {'train':[],'val':[]}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Get the inputs and labels, and send to GPU if available
            for (inputs,labels) in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the weight gradients
                optimizer.zero_grad()

                # Forward pass to get outputs and calculate loss
                # Track gradient only for training data
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model.forward(inputs).view(-1)
                    loss = criterion(outputs, labels)

                    # Backpropagation to get the gradients with respect to each weight
                    # Only if in train
                    if phase == 'train':
                        loss.backward()
                        # Update the weights
                        optimizer.step()

                # Convert loss into a scalar and add it to running_loss
                running_loss += np.sqrt(loss.item()) * labels.size(0)

            # Step along learning rate scheduler when in train
            if (phase == 'train') and (scheduler is not None):
                scheduler.step()

            # Calculate and display average loss and accuracy for the epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            costpaths[phase].append(epoch_loss)
            print('{} loss: {:.4f}'.format(phase, epoch_loss))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return costpaths


dataloaders = {'train':trainloader, 'val':valloader}
n_users = X.loc[:,'id_num'].max()+1
n_items = X.loc[:,'game_num'].max()+1
model = NNColabFiltering(n_users,n_items,embedding_dim_users=50, embedding_dim_items=50, n_activations = 100,rating_range=[0.,5.])
criterion = nn.MSELoss()
lr=0.001
n_epochs=30
wd=1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cost_paths = train_model(model,criterion,optimizer,dataloaders, device,n_epochs, scheduler=None)

def predict_rating(model,id_num,game_num, device):
    # Get predicted rating for a specific user-item pair from model
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        X = torch.Tensor([id_num,game_num]).long().view(1,-1)
        X = X.to(device)
        pred = model.forward(X)
        return pred
    
def generate_recommendations(game_rating_df,X,model,userId,device):
    # Get predicted ratings for every game
    pred_ratings = []
    for game in game_rating_df['game_num'].tolist():
        pred = predict_rating(model,userId,game,device)
        pred_ratings.append(pred.detach().cpu().item())
    # Sort gamess by predicted rating
    idxs = np.argsort(np.array(pred_ratings))[::-1]
    recs = game_rating_df.iloc[idxs]['game_num'].values.tolist()
    # Filter out games already rated by user
    game_rated = X.loc[X['id_num']==id_num, 'game_num'].tolist()
    recs = [rec for rec in recs if not rec in game_rated]
    # Filter to top 10 recommendations
    recs = recs[:10]
    # Convert games to titles
    recs_names = []
    for rec in recs:
        recs_names.append(game_rating_df.loc[game_rating_df['game_num']==rec,'title'].values[0])
    return recs_names

# Get recommendations for a random user
id_num = #ID_NUM_INPPUT#
games = game_rating_df
recs = generate_recommendations(games,X,model,id_num,device)
for i,rec in enumerate(recs):
  print('Recommendation {}: {}'.format(i,rec))