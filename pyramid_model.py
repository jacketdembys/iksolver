import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, hidden_neurons, output_dim):
        super(MLP, self).__init__()

        #self.name = "MLP [{}, {}, {}, {}]".format(str(input_dim), str(hidden_layers), str(hidden_neurons), str(output_dim))
        #self.input_dim = input_dim
        #self.hidden_size = hidden_size
        #self.output_dim = 1

        self.input_fc = nn.Linear(input_dim, hidden_neurons)
        self.hidden_fc = nn.ModuleList()
        for i in range(hidden_layers-1):
            self.hidden_fc.append(nn.Linear(hidden_neurons, hidden_neurons))
        
        self.output_fc = nn.Linear(hidden_neurons, output_dim)
        self.relu_activation = nn.ReLU()

    def forward(self, x):
        
        x = self.input_fc(x)
        x = self.relu_activation(x)  

        for i,l in enumerate(self.hidden_fc):
            x = self.hidden_fc[i](x)
            x = self.relu_activation(x)

        x = self.output_fc(x)
        x_temp = x

        return x, x_temp 
    

"""
class PyramidMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, hidden_neurons, output_dim):
        super(PyramidMLP, self).__init__()
        #self.name = "PyramidMLP [{}, {}, {}]".format(str(input_dim), str(hidden_size).replace("[","").replace("]",""), str(output_dim))
        self.name = "PyramidMLP [{}, {}, {}, {}]".format(str(input_dim), str(hidden_layers), str(hidden_neurons), str(output_dim))
                
        pyramid = []
        c_input_dim = input_dim
        for i in range(output_dim): 
            pyramid.append(MLP(c_input_dim, hidden_layers, hidden_neurons, 1))
            c_input_dim = c_input_dim + 1

    def forward(self, x):
"""
        







if __name__ == '__main__':
    
    print("\n\n")
    print("Testing MLP for Pyramid Architecture")
    input_dim = 6
    hidden_layers = 3
    hidden_neurons = 128
    output_dim = 7
    model = PMLP(input_dim, hidden_layers, hidden_neurons, output_dim)
    #print(model.name)
    print(model)

    input = torch.randn(3,6)
    output, _ = model(input)
    print(output.size())