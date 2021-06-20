import torch.nn as nn
from transformers import BertTokenizer, BertModel,BertForSequenceClassification
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import json

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
    plt.ylabel("Loss\n(Cross Entropy Loss - Decimal)", fontsize=40)
    plt.plot([i for i in range(epochs)], x, label = 'Train Loss', linewidth=3)
    plt.plot([i for i in range(epochs)], y, label = 'Valid Loss', linewidth=3)
    plt.legend(loc='upper right', prop={'size': 40}) 
    plt.show()
    plt.savefig(model_type + '_Losses')
    
    
import matplotlib.pyplot as plt
def plot_accuracy(x, y, epochs, model_type):
    plt.figure(figsize=(40,20))
    plt.title(f"Train, Valid Dataset Accuracy - {model_type}", fontsize=50)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel("Epochs", fontsize=40)
    plt.ylabel("Accuracy\n(Decimal)", fontsize=40)
    plt.plot([i for i in range(epochs)], x, label = 'Train Accuracy', linewidth=3)
    plt.plot([i for i in range(epochs)], y, label = 'Valid Accuracy', linewidth=3)
    plt.legend(loc='upper right', prop={'size': 40}) 
    plt.show()
    #plt.savefig(model_type + '_Accuracy')
    
    
#Pretrained BERT model
bert = BertModel.from_pretrained('bert-base-uncased')

class BERT(nn.Module):
    def __init__(self, bert, hidden_dimension, output_dimension, layers):
        
        super().__init__()
        
        self.bert = bert
        
        embedding_dimension = bert.config.to_dict()['hidden_size']
        
        self.rnn = nn.RNN(embedding_dimension, hidden_dimension, num_layers = layers, bidirectional = True, batch_first = True)
        
        self.linear_layer = nn.Linear(hidden_dimension * 2, output_dimension)
        
    def calculate_loss_accuracy(self, predictions, sentiment, criterion):
        
        return criterion(predictions, (sentiment).long()), calculate_accuracy(predictions, sentiment)
        
    def forward(self, text):
        
        #Ensures no gradients are calculated
        #Returns embeddings for sequence and pooled output.
        #Pooled output isn't considered as it doesn't represent good semantic summary of input
        with torch.no_grad():
            embedded = self.bert(text)[0]
       
        _, hidden = self.rnn(embedded)
        
        #Concating the forward pass and backward pass of the bidirectional behaviour
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        
        output = self.linear_layer(hidden)
        
        
        return torch.nn.functional.relu(output)
    
    def train_model(self, iterator, optimizer, criterion):
    
        epoch_loss = 0
        epoch_acc = 0

        model.train()
        i = 0
        #Uncomment to see progression if it takes too long to finish an epoch
        for batch in iterator:
            #print(f'{i}/{len(iterator)}')
            i+=1

            optimizer.zero_grad()

            predictions = self.forward(batch.Tweet).squeeze(1)

            loss, acc = self.calculate_loss_accuracy(predictions, batch.Sentiment, criterion)
            
            loss.backward()

            optimizer.step()

            epoch_loss, epoch_acc = epoch_loss + loss.item(), epoch_acc + acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate_model(self, iterator, criterion):
    
        epoch_loss = 0
        epoch_acc = 0

        model.eval()
        i = 0
        with torch.no_grad():

            for batch in iterator:
                #print(f'{i}/{len(iterator)}')
                i+=1

                predictions = self.forward(batch.Tweet).squeeze(1)

                loss, acc = self.calculate_loss_accuracy(predictions, batch.Sentiment, criterion)

                epoch_loss, epoch_acc = epoch_loss + loss.item(), epoch_acc + acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
    def execute_model(self, train_iterator, valid_iterator, optimizer, criterion, EPOCHS_T0_TRAIN_TO, model_name):
        
        loss_ = float('inf')
        
        print("===== TRAINING MODEL =====")
        #Arrays that will hold loss and accuracy values for plotting
        train_losses = []
        train_accs = []
        valid_losses = []
        valid_accs = [] 
        for epoch in range(EPOCHS_T0_TRAIN_TO):
            train_loss, train_acc = self.train_model(train_iterator, optimizer, criterion)
            valid_loss, valid_acc = self.evaluate_model(valid_iterator, criterion)

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
                "lr": 1e-3,
                "l2": 1e-3,
                "hidden_dimension":64,
                "batch_size":32
            }
            with open(os.path.join(CONTENT,"{}_{}.json".format(logs['model'],  epoch)), 'w') as f:
                 json.dump(logs, f)
               
        plot_losses(train_losses, valid_losses, EPOCHS_T0_TRAIN_TO, "BERT")
        plot_accuracy(train_accs, valid_accs, EPOCHS_T0_TRAIN_TO, "BERT")

    
    def test_model(self, model, test_iterator, criterion, model_name):
        
        print("===== TESTING MODEL =====")
        
        model.load_state_dict(torch.load(model_name))

        test_loss, test_acc = self.evaluate_model(test_iterator, criterion)

        print(f'Test Loss: {test_loss} - Test Acc: {test_acc*100:.2f}%')
