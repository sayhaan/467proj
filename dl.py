import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_df = pd.read_csv('train.csv', header=None)
dev_df = pd.read_csv('dev.csv', header=None)

column_names = [f'feature_{i}' for i in range(1, train_df.shape[1])] + ['target']
train_df.columns = column_names
dev_df.columns = column_names

X_train = train_df.drop('target', axis=1).values
y_train = train_df['target'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev = scaler.transform(dev_df.drop('target', axis=1))

X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train_split = torch.tensor(X_train_split, dtype=torch.float32).to(device)
y_train_split = torch.tensor(y_train_split, dtype=torch.float32).view(-1, 1).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.sigmoid(self.fc4(x))
        return x

input_size = X_train_split.shape[1]
model = BinaryClassifier(input_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

def calculate_accuracy(y_true, y_pred):
    predicted_classes = (y_pred >= 0.5).float()
    correct = (predicted_classes == y_true).sum().item()
    accuracy = correct / y_true.shape[0]
    return accuracy

epochs = 100
batch_size = 64
num_batches = X_train_split.shape[0] // batch_size

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    
    for i in range(0, X_train_split.shape[0], batch_size):
        X_batch = X_train_split[i:i+batch_size]
        y_batch = y_train_split[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_accuracy += calculate_accuracy(y_batch, outputs)
    
    epoch_loss /= num_batches
    epoch_accuracy /= num_batches

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        val_accuracy = calculate_accuracy(y_val, val_outputs)

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2%}, Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy:.2%}")

y_val_pred = model(X_val)
val_accuracy = calculate_accuracy(y_val, y_val_pred)
print(f"Final Validation Accuracy: {val_accuracy:.2%}")

X_dev = torch.tensor(X_dev, dtype=torch.float32).to(device)
y_dev_actual = torch.tensor(dev_df['target'].values, dtype=torch.float32).view(-1, 1).to(device)
with torch.no_grad():
    y_dev_pred = model(X_dev)

dev_accuracy = calculate_accuracy(y_dev_actual, y_dev_pred)
print(f"Dev Set Accuracy: {dev_accuracy:.2%}")

y_dev_pred_labels = (y_dev_pred >= 0.5).cpu().numpy().astype(int)
y_dev_actual_labels = y_dev_actual.cpu().numpy().astype(int)

print("\nClassification Report on Dev Set:\n", classification_report(y_dev_actual_labels, y_dev_pred_labels))
