import math
import torch
import torch.nn as nn
import torch.nn.functional as F


####################################################
# ResMLP (aggregate with Sum)
####################################################
class ResidualBlockSum(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, input_dim, output_dim):
        super(ResidualBlockSum, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, output_dim)
        #self.bn1 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(output_dim, output_dim)
        #self.bn2 = nn.BatchNorm1d(output_dim)

        # set up the shortcut  connection 
        self.shortcut = nn.Sequential()
        if input_dim != output_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                #nn.BatchNorm1d(output_dim)
            )

    def forward(self, x):
        #print("DEBUG: block input shape = {}".format(x.shape))
        residual = self.shortcut(x)

        out = self.fc1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        #out = self.bn2(out)

        out += residual
        out = self.relu(out)

        #print("DEBUG: block output shape = {}\n".format(out.shape))

        return out
    

class ResMLPSum(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, num_blocks):
        super(ResMLPSum, self).__init__()

        self.name = "ResMLP [{}, {}, {}, {}]".format(str(input_dim), str(hidden_size).replace("[","").replace("]",""), str(num_blocks), str(output_dim))
        self.input = nn.Linear(input_dim, hidden_size)
        self.blocks = nn.Sequential()
        #self.blocks.add_module('block{}'.format(0), ResidualBlock(input_dim, hidden_size))
        for i in range(num_blocks):
            self.blocks.add_module('block{}'.format(i+1), ResidualBlockSum(hidden_size, hidden_size))
        #self.blocks.add_module('block{}'.format(num_blocks-1), ResidualBlock(hidden_size, output_dim))
        self.out = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        out = self.input(x)
        out = self.blocks(out)
        out = self.out(out)

        x_temp = out

        return out, x_temp





####################################################
# ResMLP (aggregate with Concat)
####################################################
class ResidualBlockConcat(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, input_dim, output_dim):
        super(ResidualBlockConcat, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, output_dim)
        #self.bn1 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(output_dim, output_dim)
        #self.bn2 = nn.BatchNorm1d(output_dim)

        # set up the shortcut  connection 
        self.shortcut = nn.Sequential()
        if input_dim != output_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                #nn.BatchNorm1d(output_dim)
            )

    def forward(self, x):
        #print("DEBUG: block input shape = {}".format(x.shape))
        residual = self.shortcut(x)        
        #print()
        #print("x: ", x.shape)
        #print("residual: ", residual.shape)

        out = self.fc1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        #out = self.bn2(out)

        out = torch.cat([out,residual],1)
        out = self.relu(out)

        
        #print("out: ", out.shape)


        #print("DEBUG: block output shape = {}\n".format(out.shape))

        return out



class ResMLPConcat(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, num_blocks):
        super(ResMLPConcat, self).__init__()

        self.name = "ResMLPConcat [{}, {}, {}, {}]".format(str(input_dim), str(hidden_size).replace("[","").replace("]",""), str(num_blocks), str(output_dim))
        self.input = nn.Linear(input_dim, hidden_size)
        self.blocks = nn.Sequential()
        #self.blocks.add_module('block{}'.format(0), DenseBlock(input_dim, hidden_size))
        #print('block_0')
        s = 1
        for i in range(num_blocks):
            #print('block_{}'.format(i+1))
            self.blocks.add_module('block{}'.format(i+1), ResidualBlockConcat((s)*hidden_size, (s)*hidden_size))
            s *= 2
        #self.blocks.add_module('block{}'.format(num_blocks-1), ResidualBlock(hidden_size, output_dim))
        self.out = nn.Linear((s)*hidden_size, output_dim)

    def forward(self, x):
        
        out = self.input(x)
        out = self.blocks(out)
        out = self.out(out)

        #print(x.shape)

        x_temp = out

        return out, x_temp


####################################################
# DenseMLP
####################################################
class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropRate=0.0):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.droprate = dropRate
    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.fc1(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)
    


class BottleneckBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = output_dim * 4
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(input_dim, inter_planes)
        self.bn2 = nn.BatchNorm1d(inter_planes)
        self.fc2 = nn.Linear(inter_planes, output_dim)
        self.droprate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.fc1(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.fc2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)
    

class TransitionBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.droprate = dropRate
    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.fc1(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool1d(out, 2)
    


class DenseBlock(nn.Module):
    def __init__(self, nb_layers, input_dim, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, input_dim, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, input_dim, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(input_dim+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)
    


class DenseNet(nn.Module):
    def __init__(self, depth, output_dim, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0):
        super(DenseNet, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, output_dim)
        self.in_planes = in_planes

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)












class DenseBlock(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, input_dim, output_dim):
        super(DenseBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, output_dim)
        #self.bn1 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(output_dim, output_dim)
        #self.bn2 = nn.BatchNorm1d(output_dim)

        # set up the shortcut  connection 
        self.shortcut = nn.Sequential()
        if input_dim != output_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                #nn.BatchNorm1d(output_dim)
            )

    def forward(self, x):
        #print("DEBUG: block input shape = {}".format(x.shape))
        residual = self.shortcut(x)        
        print()
        print("x: ", x.shape)
        print("residual: ", residual.shape)

        out = self.fc1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        #out = self.bn2(out)

        out = torch.cat([out,residual],1)
        out = self.relu(out)

        
        print("out: ", out.shape)


        #print("DEBUG: block output shape = {}\n".format(out.shape))

        return out



class DenseMLP(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, num_blocks):
        super(DenseMLP, self).__init__()

        self.name = "DenseMLP [{}, {}, {}, {}]".format(str(input_dim), str(hidden_size).replace("[","").replace("]",""), str(num_blocks), str(output_dim))
        self.input = nn.Linear(input_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.transition12 = nn.Linear(2*hidden_size, hidden_size)
        self.transition23 = nn.Linear(3*hidden_size, hidden_size)
        self.transition34 = nn.Linear(4*hidden_size, hidden_size)
        self.transition45 = nn.Linear(5*hidden_size, hidden_size)
        self.transition5out = nn.Linear(6*hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        # input layer
        out0 = self.input(x)

        # layer block 1
        out1 = self.fc(out0)
        out1 = self.relu(out1)
        out1 = self.fc(out1)
        out01 = torch.cat([out0,out1],1)
        out01 = self.transition12(out01)

        # layer block 2
        out2 = self.fc(out01)
        out2 = self.relu(out2)
        out2 = self.fc(out2)
        out012 = torch.cat([out0,out1,out2],1)
        out012 = self.transition23(out012)

        # layer block 3
        out3 = self.fc(out012)
        out3 = self.relu(out3)
        out3 = self.fc(out3)
        out0123 = torch.cat([out0,out1,out2,out3],1)
        out0123 = self.transition34(out0123)

        # layer block 4
        out4 = self.fc(out0123)
        out4 = self.relu(out4)
        out4 = self.fc(out4)
        out01234 = torch.cat([out0,out1,out2,out3,out4],1)
        out01234 = self.transition45(out01234)

        # layer block 5
        out5 = self.fc(out01234)
        out5 = self.relu(out4)
        out5 = self.fc(out4)
        out012345 = torch.cat([out0,out1,out2,out3,out4,out5],1)
        out012345 = self.transition5out(out012345)

        # output layer
        out = self.out(out012345)

        #print(x.shape)

        x_temp = out

        return out, x_temp



if __name__ == '__main__':

    """
    print("Testing ResNet MLP (sum)")
    input_dim = 6
    hidden_size = 128
    output_dim = 7
    num_blocks = 10
    model = ResMLPSum(input_dim, hidden_size, output_dim, num_blocks)
    print(model.name)
    print(model)

    input = torch.randn(3,6)
    output, _ = model(input)
    print(output.size())
    """
    
    """
    print()
    print("Testing ResNet MLP (concat)")
    input_dim = 6
    hidden_size = 128
    output_dim = 7
    num_blocks = 5
    model = ResMLPConcat(input_dim, hidden_size, output_dim, num_blocks)
    print(model.name)
    print(model)

    input = torch.randn(3,6)
    output, _ = model(input)
    print(output.size())
    """


    
    print("\n\n")
    print("Testing DenseNet MLP")
    input_dim = 6
    hidden_size = 128
    output_dim = 7
    num_blocks = 5
    model = DenseMLP(input_dim, hidden_size, output_dim, num_blocks)
    print(model.name)
    print(model)

    input = torch.randn(3,6)
    output, _ = model(input)
    print(output.size())
    
    
