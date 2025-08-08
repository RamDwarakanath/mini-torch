import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

epochs = 500
lr = 0.1

model = nn.Sequential(nn.Linear(in_features=1, out_features=2),
                       nn.Tanh(),
                       nn.Linear(in_features=2, out_features=1)
                       )

optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.MSELoss()

X = torch.tensor(np.array([[0], [1], [2]]), dtype=torch.float32) #do i always need to put this for the data type
y = torch.tensor(np.array([[0], [1], [2]]), dtype=torch.float32)

print(f"Initial Predictions, ", model(X))

# Training Loop
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

    # print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
    print(f"Epoch: {epoch + 1}, Loss: {loss}")

print(f"New Predictions, ", model(X))