from data import dataset,train_loader,test_loader
import torch.nn as nn
import torch.functional as F

# Instantiate class names
class_names = dataset.classes

# Build convolutional neural network model architecture
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        self.fc1 = nn.Linear(16*54*54,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,20)
        self.fc4 = nn.Linear(20,len(class_names))
        
     def forward(self,X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2D(X,2,2)
        X = X.view(-1,16*54*54)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        
        return F.log_softmax(X,dim=1)

# Initialize model
model = CNN()
# Initialize loss
criterion = nn.CrossEntropyLoss()
# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

start_time = time.time()
train_losses = []
test_losses = []
train_correct = []
test_correct = []

# Instantiate number of epochs
epochs = 10

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0
    for b, (X_train,y_train) in enumerate(train_loader):
        b += 1

        y_pred = model(X_train)

        loss = criterion(y_pred,y_train)

        predicted = torch.max(y_pred.data,1)[1]
        batch_corr = (predicted==y_train).sum()
        trn_corr += batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if b%5==0:
            print(f"epoch: {i} loss: {loss.item} batch: {b} accuracy: {trn_corr.item()*100/(10*b):7.3f}%")
        
        loss = loss.detach().numpy()
        train_losses.append(loss)
        train_correct.append(trn_corr)
        
        with torch.no_grad():
            for b, (X_test,y_test) in enumerate(test_loader):
                    y_val = model(X_test)
                    loss = criterion(y_val,y_test)
                    predicted = torch.max(y_val.data,1)[1]
                    btach_corr = (predicted==y_test).sum()
                    tst_corr += btach_corr
                loss = loss.detach().numpy()
                test_losses.append(loss)
                test_correct.append(tst_corr)

print(f"\nDuration: {time.time() - start_time:.0f} seconds")

plt.plot(train_losses,label="Train Loss")
plt.plot(test_losses,label="Test Loss")
