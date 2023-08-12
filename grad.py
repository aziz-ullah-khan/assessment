import torch

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

input_sample, input_feature = X.size()

#model = torch.nn.Linear(in_features=input_feature, out_features=input_feature)

class LinearRegression(torch.nn.Module):
    def __init__(self, input_feature, output_feature):
        super(LinearRegression, self).__init__()
        self.lin = torch.nn.Linear(input_feature, output_feature)
    
    def forward(self, X):
        return self.lin(X)

model = LinearRegression(input_feature, input_feature)

loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

n_itr = 5000

for epoch in range(n_itr):
    y_pred = model(X)

    l = loss(Y, y_pred)

    l.backward()

    optimizer.step()

    optimizer.zero_grad()

    print(f"Predicted: {y_pred}")
