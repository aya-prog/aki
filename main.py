import torch
import pandas as pd
import matplotlib.pyplot as plt

# Laden des Datensatzes
df = pd.read_csv(r'C:\Users\Aya\Desktop\Exercise 1\winequality-white.csv', sep=';')
inputs = torch.tensor(df.iloc[:, 0:11].values, dtype=torch.float32)
labels = torch.tensor(df.iloc[:, 11:12].values, dtype=torch.float32)

# Definition der Netzwerkarchitektur
class WineQualityNN(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, num_hidden_nodes):
        super(WineQualityNN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(num_inputs, num_hidden_nodes))
        for i in range(num_hidden_layers - 1):
            self.layers.append(torch.nn.Linear(num_hidden_nodes, num_hidden_nodes))
        self.layers.append(torch.nn.Linear(num_hidden_nodes, num_outputs))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = torch.softmax(self.layers[-1](x), dim=1)
        return x

# Definieren der Hyperparameter
num_inputs = 11
num_outputs = 1
num_hidden_layers = 2
num_hidden_nodes = 3
learning_rate = 0.001
num_epochs = 3000
batch_size = 64

# Initialisieren des Modells und Optimizers
model = WineQualityNN(num_inputs, num_outputs, num_hidden_layers, num_hidden_nodes)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

# Trainieren des Modells
losses = []
for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0
    for i in range(0, inputs.shape[0], batch_size):
        batch_inputs = inputs[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1
    losses.append(epoch_loss / num_batches)

# Plotting des Losses über die Epochen
plt.plot(range(num_epochs), losses)
plt.xlabel('Epochen')
plt.ylabel('Verlust')
plt.title('Trainingsverlust')
plt.show()