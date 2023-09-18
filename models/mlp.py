import torch
import torch.nn as nn
import ipdb

# Define the fully connected network
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()  

        # Add input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.ELU())

        # Add hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ELU())

        # Add output layer
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if 1:
    view_train_kwargs_lookup = {
        0 : dict(input_size=1, hidden_size=8, output_size=1024, num_layers=3),
        1 : dict(input_size=1, hidden_size=32, output_size=1024, num_layers=3),
        2 : dict(input_size=1, hidden_size=128, output_size=1024, num_layers=3),
        3 : dict(input_size=1, hidden_size=8, output_size=1024, num_layers=3),
    }
else: 
    # delete this later 
    view_train_kwargs_lookup = {
        3 : dict(input_size=1, hidden_size=8, output_size=1024, num_layers=2),
        1 : dict(input_size=1, hidden_size=8, output_size=1024, num_layers=3),
        0 : dict(input_size=1, hidden_size=32, output_size=1024, num_layers=3),
        3 : dict(input_size=1, hidden_size=128, output_size=1024, num_layers=3),
    }

if __name__=="__main__":
    kwargs = dict(input_size=1, hidden_size=128, output_size=1024, num_layers=3)
    model = MLP(**kwargs)
    print(model)
    ipdb.set_trace()
    pass