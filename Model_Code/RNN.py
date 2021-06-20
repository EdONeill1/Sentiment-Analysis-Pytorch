import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import json
import torch.optim as optim
from torchtext import data
from string import punctuation
from nltk.corpus import stopwords
import random 

import nltk

def calculate_accuracy(preds, y):
    top_pred = preds.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
  
    return acc

import matplotlib.pyplot as plt
def plot_losses(x, y, epochs, model_type):
    plt.figure(figsize=(40,20))
    plt.title(f"Train, Valid Dataset Loss - {model_type}", fontsize=50)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel("Epochs", fontsize=40)
    plt.ylabel("Loss\n(Cross Entropy Loss)", fontsize=40)
    plt.plot([i for i in range(epochs)], x, label = 'Train Loss', linewidth=3)
    plt.plot([i for i in range(epochs)], y, label = 'Valid Loss', linewidth=3)
    plt.legend(loc='upper right', prop={'size': 40}) 
    plt.show()
    #plt.savefig(model_type + '_Losses')
    
    
import matplotlib.pyplot as plt
def plot_accuracy(x, y, epochs, model_type):
    plt.figure(figsize=(40,20))
    plt.title(f"Train, Valid Dataset Accuracy - {model_type}", fontsize=50)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel("Epochs", fontsize=40)
    plt.ylabel("Accuracy\n(Percentage)", fontsize=40)
    plt.plot([i for i in range(epochs)], x, label = 'Train Accuracy', linewidth=3)
    plt.plot([i for i in range(epochs)], y, label = 'Valid Accuracy', linewidth=3)
    plt.legend(loc='upper right', prop={'size': 40}) 
    plt.show()
    #plt.savefig(model_type + '_Accuracy')

class RNN(nn.Module):

    def __init__(self, INPUT_DIMENSION, EMBEDDING_DIMENSION, HIDDEN_DIMENSION, OUTPUT_DIMENSION):
        
        super().__init__()
        
        self.embedding = nn.Embedding(INPUT_DIMENSION, EMBEDDING_DIMENSION)
        
        self.rnn = nn.RNN(EMBEDDING_DIMENSION, HIDDEN_DIMENSION)
        
        self.linear_layer = nn.Linear(HIDDEN_DIMENSION, OUTPUT_DIMENSION, bias = True)
       
    
    
    # Embed text -> apply it to rnn -> apply it to linear layer -> apply activation function
    def forward(self, tweet, tweet_lengths):
        
        embedded = self.embedding(tweet)
        
        output, hidden = self.rnn(embedded)
       
        l1 = self.linear_layer(hidden.squeeze(0))

        return torch.relu(l1)
    
    
    def calculate_loss_accuracy(self, predictions, sentiment, criterion):
        
        return criterion(predictions, (sentiment).long()), calculate_accuracy(predictions, sentiment)
    
    
    def train_model(self, model, iterator, optimizer, criterion):
    
        epoch_loss, epoch_acc = 0, 0
    
        model.train()
    
        for batch in iterator:
        
            optimizer.zero_grad()

            tweet, tweet_len = batch.Tweet

            predictions = model(tweet, tweet_len)
            
            loss, acc = self.calculate_loss_accuracy(model(tweet, tweet_len), batch.Sentiment, criterion)

            loss.backward()

            optimizer.step()

            epoch_loss, epoch_acc = epoch_loss + loss.item(), epoch_acc + acc.item()
        
        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
    
    def evaluate_model(self, model, iterator, criterion):
        epoch_loss, epoch_acc = 0, 0
        model.eval()
    
        with torch.no_grad():
            for batch in iterator:

                tweet, tweet_len = batch.Tweet

                predictions = model(tweet, tweet_len)

                loss, acc = self.calculate_loss_accuracy(model(tweet, tweet_len), batch.Sentiment, criterion)

                epoch_loss, epoch_acc = epoch_loss + loss.item(), epoch_acc + acc.item()
        
        return epoch_loss / len(iterator), epoch_acc / len(iterator)   
    
    def execute_model(self, model, train_iterator, valid_iterator, optimizer, criterion, epochs, lr, model_name):
        
        loss_ = float('inf')
        
        print("===== TRAINING MODEL =====")
        #Arrays that will hold loss and accuracy values for plotting
        train_losses = []
        train_accs = []
        valid_losses = []
        valid_accs = [] 
        

        for epoch in range(epochs):

            train_loss, train_acc = self.train_model(model, train_iterator, optimizer, criterion)
            valid_loss, valid_acc = self.evaluate_model(model, valid_iterator, criterion)

            #If the validation loss is less than loss_, loss_ becomes validation loss
            # representing the current best loss in the model.
            if valid_loss < loss_:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), model_name)

            print(f'Epoch : {epoch}')
            print(f'Train Loss: {train_loss} - Train Acc: {train_acc*100:.2f}%')
            print(f'Valid Loss: {valid_loss} - Valid Acc: {valid_acc*100:.2f}%')
            
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)
            train_accs.append(train_acc)
            
            logs ={
                "model": model_name,
                "train_losses": train_losses,
                "train_accs": train_accs,
                "val_losses": valid_losses,
                "val_accs": valid_accs,
                "best_val_epoch": int(np.argmax(valid_accs)+1),
                "model": model_name,
                "lr": lr,
                "l2": 1e-3,
                "max_size_vocab":5000,
                "hidden_dimension":64,
                "batch_size": 64
            }
            with open(os.path.join(PATH, "{}_{}.json".format(logs['model'],  epoch)), 'w') as f:
                json.dump(logs, f)


        plot_losses(train_losses, valid_losses, epochs, "RNN")
        plot_accuracy(train_accs, valid_accs, epochs, "RNN")
            
        
    def test_model(self, model, test_iterator, criterion, model_name):
        
        print("===== TESTING MODEL =====")
        
        model.load_state_dict(torch.load(model_name))

        test_loss, test_acc = self.evaluate_model(model, test_iterator, criterion)

        print(f'Test Loss: {test_loss*100:.3f} - Test Acc: {test_acc*100:.3f}%')
