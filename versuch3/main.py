import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the sequence
seq = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6]])

# Define the model
class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.U_f = nn.Parameter(torch.Tensor(2, 2))
        self.V_f = nn.Parameter(torch.Tensor(2, 2))
        self.b_f = nn.Parameter(torch.Tensor(1, 2))
        self.U_i = nn.Parameter(torch.Tensor(2, 2))
        self.V_i = nn.Parameter(torch.Tensor(2, 2))
        self.b_i = nn.Parameter(torch.Tensor(1, 2))
        self.U_o = nn.Parameter(torch.Tensor(2, 2))
        self.V_o = nn.Parameter(torch.Tensor(2, 2))
        self.b_o = nn.Parameter(torch.Tensor(1, 2))
        self.U_g = nn.Parameter(torch.Tensor(2, 2))
        self.V_g = nn.Parameter(torch.Tensor(2, 2))
        self.b_g = nn.Parameter(torch.Tensor(1, 2))
        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.1, 0.1)

    def forward(self, input, hx=None):
        h_t, c_t = hx or (torch.zeros(1, 2), torch.zeros(1, 2))
        output_seq = []
        for x_t in input:
            f_t = torch.sigmoid(torch.matmul(x_t, self.U_f) + torch.matmul(h_t, self.V_f) + self.b_f)
            i_t = torch.sigmoid(torch.matmul(x_t, self.U_i) + torch.matmul(h_t, self.V_i) + self.b_i)
            o_t = torch.sigmoid(torch.matmul(x_t, self.U_o) + torch.matmul(h_t, self.V_o) + self.b_o)
            g_t = torch.tanh(torch.matmul(x_t, self.U_g) + torch.matmul(h_t, self.V_g) + self.b_g)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            output_seq.append(h_t)
        return torch.stack(output_seq, dim=1)

# Define the training function
def train(net, seq):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    for epoch in range(20):
        optimizer.zero_grad()
        input_seq = torch.Tensor(seq).unsqueeze(0)
        output_seq = net(input_seq)
        target_seq = torch.Tensor(seq).unsqueeze(0)
        loss = criterion(output_seq, target_seq)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Train the model and print the final output sequence
net = LSTMNet()
train(net, seq)
input_seq = torch.Tensor(seq).unsqueeze(0)
output_seq = net(input_seq)
print(f'Input Sequence: \n{seq}')
print(f'Output Sequence: \n{output_seq.detach().numpy()[0]}')

