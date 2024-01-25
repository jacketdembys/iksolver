import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import count_parameters

class DenseBlock(nn.Module):
    def __init__(self, in_features, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = self.make_dense_layer(in_features + i * growth_rate, growth_rate)
            self.layers.append(layer)

    def make_dense_layer(self, in_features, growth_rate):
        return nn.Sequential(
            #nn.BatchNorm1d(in_features),
            #nn.ReLU(inplace=True),
            nn.Linear(in_features, growth_rate),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)

class TransitionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(TransitionLayer, self).__init__()
        self.layer = nn.Sequential(
            #nn.BatchNorm1d(in_features),
            #nn.ReLU(inplace=True),
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)

class DenseNet(nn.Module):
    def __init__(self, in_features, growth_rate, block_config, out_features):
        super(DenseNet, self).__init__()
        self.name = "DenseMLP3 [{}, {}, {}, {}]".format(str(in_features), str(growth_rate), str(len(block_config)), str(out_features))
        self.input = nn.Linear(in_features, growth_rate)
        self.input_relu = nn.ReLU(inplace=True)
        self.features = nn.Sequential()
        in_features = growth_rate
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(in_features, growth_rate, num_layers)
            self.features.add_module(f'denseblock_{i+1}', block)
            in_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                """
                trans = TransitionLayer(in_features, in_features // 2)
                self.features.add_module(f'transition_{i+1}', trans)
                in_features = in_features // 2
                """
                trans = TransitionLayer(in_features, growth_rate)
                self.features.add_module(f'transition_{i+1}', trans)
                in_features = growth_rate

        self.classifier = nn.Linear(in_features, out_features)

    def forward(self, x):
        out = self.input(x)
        out = self.input_relu(out)
        out = self.features(out)
        #out = F.adaptive_avg_pool1d(out.unsqueeze(-1), 1)
        #out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out2 = out
        return out, out2

if __name__ == '__main__':
    # Example usage:
    in_features = 6  # Input features for a flattened image
    growth_rate = 1024
    block_config = [2,2,2,2,2,2] # [2, 2, 2, 2]  # Number of layers in each dense block
    out_features = 7  # Number of output classes

    model = DenseNet(in_features, growth_rate, block_config, out_features)
        
    print(model)
    print("==> Trainable parameters: {}".format(count_parameters(model)))
