import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

class TrainingPipeline ():
    def __init__(self,model,data,batch_size=32,lr =0.01,epochs=100, optimizer = optim.Adam, criterion = nn.BCELoss, device = None, train_percent = 0.8):
        '''
        Initialize the Training Pipeline
        Args:
            :param model: PASS IN MODEL CLASS.
            :param data: pass into data in tensor form.
            :param batch_size: batch size for training.
            :param lr: learning rate.
            :param epochs: Number of training epochs.
            :param device: The device model and data will be passed into(ie cpu or cuda).
            :param optimizer: PASS IN THE OPTIMIZER CLASS.
            :param criterion: PASS IN LOSS FUNCTION CLASS.
            :param train_percent: used to split into training and validation 
        '''
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.criterion = criterion()
        self.optimizer = optimizer(self.model.parameters(),lr = lr)
        self.epochs = epochs

        #train test split(default 8:2)
        train_size = int(train_percent* len(data))
        val_size = len(data) - train_size
        train_data , val_data = random_split(data,[train_size,val_size])

        self.train_loader = DataLoader(train_data,batch_size = batch_size,shuffle = True)
        self.val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False)



    def train_one_epoch(self):
        '''
        Runs one training loop through the model using training data
        Returns:
            float: Average loss for the batch
        '''
        self.model.train()
        total_loss = 0
        #loops through each batch
        for xb,yb in self.train_loader:
            xb , yb = xb.to(self.device) , yb.to(self.device)
            preds = self.model(xb)
            loss = self.criterion(preds,yb)
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return total_loss / len(self.train_loader) #returns average loss for the batch
            

    def validate(self):
        '''
        Evaluates the current model using testing data
        Returns:
            float: average loss for the batch
        '''
        self.model.eval()
        total_loss = 0
        with torch.inference_mode():
            #loops through each batch
            for xb,yb in self.val_loader:
                xb , yb = xb.to(self.device) , yb.to(self.device)
                preds = self.model(xb)
                loss = self.criterion(preds,yb)
                total_loss += loss.item()

                
        return total_loss / len(self.val_loader) #returns average loss for the batch

    def training(self):
        '''
        Loop where all the training actually happens, prints out training and validation loss
        '''
        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()
            if(epoch%5 == 0):
                print(f"Train loss: {train_loss}    Validation loss:{val_loss}")

    def save(self,path):
        '''
        Saves the model to the given path
        Args:
            :param path: path where model is saved
        
        '''
        torch.save(self.model.state_dict(),path)

