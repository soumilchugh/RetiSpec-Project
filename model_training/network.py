import torch 
import torch.nn as nn
import torch.nn.functional as F 

class Conv(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride = 1):
        super(Conv, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, kernel_size//2),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.maxpool(self.conv(x))


class FC(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FC, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_channels, output_channels),
            nn.BatchNorm1d(output_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(x)


class Output(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Output, self).__init__()

        self.output = nn.Sequential(
            nn.Linear(input_channels, output_channels),
        )

    def forward(self, x):
        return self.output(x)


class Network(nn.Module):

    def __init__(self, config):

        super(Network, self).__init__()

        self.rgb_feature_extractor = nn.Sequential(
            Conv(input_channels = 3, output_channels=8, kernel_size= 3),
            Conv(input_channels = 8, output_channels=16, kernel_size= 3),
            Conv(input_channels = 16, output_channels=32, kernel_size= 3)
        )

        self.ir_feature_extractor = nn.Sequential(
            Conv(input_channels = 1, output_channels=8, kernel_size= 3),
            Conv(input_channels = 8, output_channels=16, kernel_size= 3),
            Conv(input_channels = 16, output_channels=32, kernel_size= 3)
        )


        self.mlp = nn.Sequential(
            FC(input_channels = 8*8*32, output_channels=128),
            FC(input_channels = 128, output_channels=32),
            
        )

        self.output_layer = Output(input_channels=32, output_channels=1)
                    
        self._initialize_weight()

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        

    def forward(self, img, img_ni):

        rgb_features = self.rgb_feature_extractor(img)
        ir_features = self.ir_feature_extractor(img_ni)
        rgb_flatten_layer = rgb_features.view(rgb_features.size(0),-1)
        ir_flatten_layer = ir_features.view(ir_features.size(0),-1)
        mlp = self.mlp(rgb_flatten_layer*ir_flatten_layer)
        out = self.output_layer(mlp)
        
        return out, torch.sigmoid(out)

