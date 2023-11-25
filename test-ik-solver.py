import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

        # set up the shortcut  connection 
        self.shortcut = nn.Sequential()
        if input_dim != output_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim)
            )

    def forward(self, x):
        print("DEBUG: block input shape = {}".format(x.shape))
        residual = self.shortcut(x)

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        print("DEBUG: block output shape = {}\n".format(out.shape))

        return out
    

class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, num_blocks):
        super(ResNet, self).__init__()

        
        self.blocks = nn.Sequential()
        self.blocks.add_module('block{}'.format(0), ResidualBlock(input_dim, hidden_size))
        for i in range(num_blocks-1):
            self.blocks.add_module('block{}'.format(i+1), ResidualBlock(hidden_size, hidden_size))
        #self.blocks.add_module('block{}'.format(num_blocks-1), ResidualBlock(hidden_size, output_dim))
        self.out = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        out = self.blocks(x)
        out = self.out(out)

        return out
    
if __name__ == '__main__':
    model = ResNet(6,128,7,3)
    print(model)

    input = torch.randn(3,6)
    output = model(input)
    print(output.size())
