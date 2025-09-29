import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset

class TrainingPipeline ():
    def __init__(self,model,data,batch_size=32,lr =0.01,epochs=100,device = None):
        self.device = device or torch.device("cuda" if torch.cuda_is_available() else "cpu")
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(),lr = lr)
        self.epochs = epochs

        #train test split(0.8:0.2)
        train_size = int(len(data))
        val_size = len(data) - train_size
        train_data , val_data = random_split(data,[train_size,val_size])

        self.train_loader = DataLoader(train_data,batch_size = batch_size,shuffle = True)
        self.val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False)



    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        #loops through each batch
        for xb,yb in self.train_loader:
            xb , yb = xb.to(self.device) , yb.to(self.device)
            preds = self.model.predict(xb)
            loss = self.criterion(yb,preds)
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return total_loss / len(self.train_loader) #returns average loss for the batch
            

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.inference_mode:
            #loops through each batch
            for xb,yb in self.val_data:
                xb , yb = xb.to(self.device) , yb.to(self.device)
                preds = self.model.predict(xb)
                loss = self.criterion(yb,preds)
                total_loss += loss.item()

                
        return total_loss / len(self.val_loader) #returns average loss for the batch

    def training(self):
        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()
            if(epoch%5 == 0):
                print(f"Train loss: {train_loss}    Validation loss:{val_loss}")

    def save(self,path):
        torch.save(self.model.state_dict(),path)

        