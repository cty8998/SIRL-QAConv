import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random
import torchvision.transforms as transforms

import torchvision

from PIL import Image
import os

# from .utils import *

def calc_mean_std(input, eps=1e-5):
    batch_size, channels = input.shape[:2]

    reshaped = input.view(batch_size, channels, -1) # Reshape channel wise
    mean = torch.mean(reshaped, dim = 2).view(batch_size, channels, 1, 1) # Calculat mean and reshape
    std = torch.sqrt(torch.var(reshaped, dim=2)+eps).view(batch_size, channels, 1, 1) # Calculate variance, add epsilon (avoid 0 division),
                                                                                      # calculate std and reshape
    return mean, std


def AdaIn(content, style):
    # assert content.shape[:2] == style.shape[:2] # Only first two dim, such that different image sizes is possible
    batch_size, n_channels = content.shape[:2]
    mean_content, std_content = calc_mean_std(content)
    mean_style, std_style = calc_mean_std(style)

    output = std_style*((content - mean_content) / (std_content)) + mean_style # Normalise, then modify mean and std
    return output

def AdaIn_for(content, style, mask_c, mask_s, eps=1e-5):
    batch_size, n_channels = content.shape[:2]

    mean_c_list = []
    std_c_list = []
    mean_s_list = []
    std_s_list = []
    for k in range(batch_size):
        index_c = torch.nonzero(mask_c[k].cpu()) 
        content_for = content[k][:,index_c[:,1],index_c[:,2]]
        index_s = torch.nonzero(mask_s[k].cpu()) 
        style_for = style[k][:,index_s[:,1],index_s[:,2]]
        mean_c = torch.mean(content_for.view(n_channels,-1), dim = -1).view(1, n_channels, 1, 1) 
        std_c = torch.sqrt(torch.var(content_for.view(n_channels,-1), dim=-1)+eps).view(1, n_channels, 1, 1)
        mean_s = torch.mean(style_for.view(n_channels,-1), dim = -1).view(1, n_channels, 1, 1) 
        std_s = torch.sqrt(torch.var(style_for.view(n_channels,-1), dim=-1)+eps).view(1, n_channels, 1, 1)
        mean_c_list.append(mean_c)
        std_c_list.append(std_c)
        mean_s_list.append(mean_s)
        std_s_list.append(std_s)

    mean_content = torch.cat(mean_c_list, dim=0)
    std_content = torch.cat(std_c_list, dim=0)
    mean_style = torch.cat(mean_s_list, dim=0)
    std_style = torch.cat(std_s_list, dim=0)

    output = std_style*((content - mean_content) / (std_content)) + mean_style 
    output = content * (1 - mask_c) + output * mask_c
    return output

def AdaIn_back(content, style, mask_c, mask_s, eps=1e-5):
    mask_c = 1 - mask_c
    mask_s = 1 - mask_s
    batch_size, n_channels = content.shape[:2]

    mean_c_list = []
    std_c_list = []
    mean_s_list = []
    std_s_list = []
    for k in range(batch_size):
        index_c = torch.nonzero(mask_c[k].cpu()) 
        content_back = content[k][:,index_c[:,1],index_c[:,2]]
        mean_c = torch.mean(content_back.view(n_channels,-1), dim = -1).view(1, n_channels, 1, 1) 
        std_c = torch.sqrt(torch.var(content_back.view(n_channels,-1), dim=-1)+eps).view(1, n_channels, 1, 1)
        mean_c_list.append(mean_c)
        std_c_list.append(std_c)

        index_s = torch.nonzero(mask_s[k].cpu()) 
        style_back = style[k][:,index_s[:,1],index_s[:,2]]
        mean_s = torch.mean(style_back.view(n_channels,-1), dim = -1).view(1, n_channels, 1, 1) 
        std_s = torch.sqrt(torch.var(style_back.view(n_channels,-1), dim=-1)+eps).view(1, n_channels, 1, 1)
        mean_s_list.append(mean_s)
        std_s_list.append(std_s)

    mean_content = torch.cat(mean_c_list, dim=0)
    std_content = torch.cat(std_c_list, dim=0)
    mean_style = torch.cat(mean_s_list, dim=0)
    std_style = torch.cat(std_s_list, dim=0)

    output = std_style*((content - mean_content) / (std_content)) + mean_style 
    output = content * (1 - mask_c) + output * mask_c
    return output

def Content_loss(input, target): # Content loss is a simple MSE Loss
    loss = F.mse_loss(input, target)
    return loss

def Style_loss(input, target):
    mean_loss, std_loss = 0, 0

    for input_layer, target_layer in zip(input, target): 
        mean_input_layer, std_input_layer = calc_mean_std(input_layer)
        mean_target_layer, std_target_layer = calc_mean_std(target_layer)

        mean_loss += F.mse_loss(mean_input_layer, mean_target_layer)
        std_loss += F.mse_loss(std_input_layer, std_target_layer)

    return mean_loss+std_loss

def Style_loss_for(input, target, mask_c, eps=1e-5):
    mask_c = mask_c
    mean_loss, std_loss = 0, 0
    input_layer = input[-1]
    target_layer = target[-1]
    batch_size, n_channels = input_layer.shape[:2]

    mean_input_layer_list = []
    std_input_layer_list = []
    mean_target_layer_list = []
    std_target_layer_list = []
    for k in range(batch_size):
        index_c = torch.nonzero(mask_c[k]) 
        input_layer_for = input_layer[k][:,index_c[:,1],index_c[:,2]]
        target_layer_for = target_layer[k][:,index_c[:,1],index_c[:,2]]
        mean_input_layer = torch.mean(input_layer_for.view(n_channels,-1), dim = -1).view(1, n_channels, 1, 1) 
        std_input_layer = torch.sqrt(torch.var(input_layer_for.view(n_channels,-1), dim=-1)+eps).view(1, n_channels, 1, 1)
        mean_target_layer = torch.mean(target_layer_for.view(n_channels,-1), dim = -1).view(1, n_channels, 1, 1) 
        std_target_layer = torch.sqrt(torch.var(target_layer_for.view(n_channels,-1), dim=-1)+eps).view(1, n_channels, 1, 1)
        mean_input_layer_list.append(mean_input_layer)
        std_input_layer_list.append(std_input_layer)
        mean_target_layer_list.append(mean_target_layer)
        std_target_layer_list.append(std_target_layer)
    mean_input = torch.cat(mean_input_layer_list, dim=0)
    std_input = torch.cat(std_input_layer_list, dim=0)
    mean_target = torch.cat(mean_target_layer_list, dim=0)
    std_target = torch.cat(std_target_layer_list, dim=0)

    mean_loss = F.mse_loss(mean_input, mean_target)
    std_loss = F.mse_loss(std_input, std_target)

    return mean_loss+std_loss

def Style_loss_back(input, target, mask_c, eps=1e-5):
    mask_c = 1 - mask_c
    mean_loss, std_loss = 0, 0
    input_layer = input[-1]
    target_layer = target[-1]
    batch_size, n_channels = input_layer.shape[:2]

    mean_input_layer_list = []
    std_input_layer_list = []
    mean_target_layer_list = []
    std_target_layer_list = []
    for k in range(batch_size):
        index_c = torch.nonzero(mask_c[k]) 
        input_layer_back = input_layer[k][:,index_c[:,1],index_c[:,2]]
        target_layer_back = target_layer[k][:,index_c[:,1],index_c[:,2]]
        mean_input_layer = torch.mean(input_layer_back.view(n_channels,-1), dim = -1).view(1, n_channels, 1, 1) 
        std_input_layer = torch.sqrt(torch.var(input_layer_back.view(n_channels,-1), dim=-1)+eps).view(1, n_channels, 1, 1)
        mean_target_layer = torch.mean(target_layer_back.view(n_channels,-1), dim = -1).view(1, n_channels, 1, 1) 
        std_target_layer = torch.sqrt(torch.var(target_layer_back.view(n_channels,-1), dim=-1)+eps).view(1, n_channels, 1, 1)
        mean_input_layer_list.append(mean_input_layer)
        std_input_layer_list.append(std_input_layer)
        mean_target_layer_list.append(mean_target_layer)
        std_target_layer_list.append(std_target_layer)
    mean_input = torch.cat(mean_input_layer_list, dim=0)
    std_input = torch.cat(std_input_layer_list, dim=0)
    mean_target = torch.cat(mean_target_layer_list, dim=0)
    std_target = torch.cat(std_target_layer_list, dim=0)

    mean_loss = F.mse_loss(mean_input, mean_target)
    std_loss = F.mse_loss(std_input, std_target)

    return mean_loss+std_loss

# The style transfer network
class StyleTransferNetwork(nn.Module):
    def __init__(self,
                device, # "cpu" for cpu, "cuda" for gpu
                learning_rate=1e-4,
                learning_rate_decay=5e-5, # Decay parameter for the learning rate
                gamma=2.0, # Controls importance of StyleLoss vs ContentLoss, Loss = gamma*StyleLoss + ContentLoss
                train=True, # Wether or not network is training
                load_fromstate=False, # Load from checkpoint?
                load_path=None # Path to load checkpoint
                ):
        super().__init__()

        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.gamma = gamma

        self.encoder = Encoder(device) # A pretrained vgg19 is used as the encoder
        self.decoder = Decoder().to(device)

        self.optimiser = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
        self.iters = 0

        if load_fromstate:
            state = torch.load(load_path, map_location=torch.device('cpu'))
            self.decoder.load_state_dict(state["Decoder"])
            self.optimiser.load_state_dict(state["Optimiser"])
            self.iters = state["iters"]


    def set_train(self, boolean): # Change state of network
        assert type(boolean) == bool
        self.train = boolean

    def adjust_learning_rate(self, optimiser, iters): # Simple learning rate decay
        lr = self.learning_rate / (1.0 + self.learning_rate_decay * iters)
        for param_group in optimiser.param_groups:
            param_group['lr'] = lr

    def forward(self, content, style, mask_c, mask_s, generate_id=1, alpha=0.2): # Alpha can be used while testing to control the importance of the transferred style
        
        mask_c = torch.nn.functional.upsample(mask_c, scale_factor=0.125, mode='nearest')
        mask_s = torch.nn.functional.upsample(mask_s, scale_factor=0.125, mode='nearest')
        # Encode style and content
        layers_style = self.encoder(style, True) # if train: returns all states
        layer_content = self.encoder(content, False) # for the content only the last layer is important

        if generate_id==0:
            style_applied = AdaIn(layer_content, layers_style[-1])
            style_applied_upscaled = self.decoder(style_applied)

            layers_style_applied = self.encoder(style_applied_upscaled, True)
            # content_loss = Content_loss(layers_style_applied[-1], layer_content)
            # style_loss = Style_loss(layers_style_applied, layers_style)
            # loss_comb = content_loss + self.gamma*style_loss

            return layer_content, layers_style, layers_style_applied, style_applied_upscaled

        elif generate_id==1:
            style_applied = AdaIn(layer_content, layers_style[-1])
            style_applied_upscaled = self.decoder(style_applied)
            style_applied_back = AdaIn_back(layer_content, layers_style[-1], mask_c, mask_s)
            style_applied_upscaled_back = self.decoder(style_applied_back)

            layers_style_applied = self.encoder(style_applied_upscaled, True)
            layers_style_applied_back = self.encoder(style_applied_upscaled_back, True)
            content_loss = (Content_loss(layers_style_applied[-1], layer_content) + Content_loss(layers_style_applied_back[-1] * (1-mask_c), layer_content * (1-mask_c))) / 2
            style_loss = (Style_loss(layers_style_applied, layers_style) + Style_loss_back(layers_style_applied_back, layers_style, mask_c)) / 2
            loss_comb = content_loss + self.gamma*style_loss

            return loss_comb, style_applied_upscaled, style_applied_upscaled_back
        
        elif generate_id==2:
            style_applied = AdaIn(layer_content, layers_style[-1])
            style_applied_upscaled = self.decoder(style_applied)
            style_applied_for = AdaIn_for(layer_content, layers_style[-1], mask_c, mask_s)
            style_applied_upscaled_for = self.decoder(style_applied_for)
            # style_applied_back = AdaIn_back(layer_content, layers_style[-1], mask_c, mask_s)
            # style_applied_upscaled_back = self.decoder(style_applied_back)

            layers_style_applied = self.encoder(style_applied_upscaled, True)
            layers_style_applied_for = self.encoder(style_applied_upscaled_for, True)
            # layers_style_applied_back = self.encoder(style_applied_upscaled_back, True)
            # content_loss = (Content_loss(layers_style_applied[-1], layer_content) + Content_loss(layers_style_applied_for[-1] * mask_c, layer_content * mask_c) + Content_loss(layers_style_applied_back[-1] * (1-mask_c), layer_content * (1-mask_c))) / 3
            # style_loss = (Style_loss(layers_style_applied, layers_style) + Style_loss_for(layers_style_applied_for, layers_style, mask_c) + Style_loss_back(layers_style_applied_back, layers_style, mask_c)) / 3
            content_loss = (Content_loss(layers_style_applied[-1], layer_content) + Content_loss(layers_style_applied_for[-1] * mask_c, layer_content * mask_c)) / 2
            style_loss = (Style_loss(layers_style_applied, layers_style) + Style_loss_for(layers_style_applied_for, layers_style, mask_c)) / 2
            loss_comb = content_loss + self.gamma*style_loss

            return loss_comb, style_applied_upscaled, style_applied_upscaled_for

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.padding = nn.ReflectionPad2d(padding=1) # Using reflection padding as described in vgg19
        self.UpSample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv4_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0)

        self.conv3_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.conv3_4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0)

        self.conv2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.conv1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=0)


    def forward(self, x):
        out = self.UpSample(F.relu(self.conv4_1(self.padding(x))))

        out = F.relu(self.conv3_1(self.padding(out)))
        out = F.relu(self.conv3_2(self.padding(out)))
        out = F.relu(self.conv3_3(self.padding(out)))
        out = self.UpSample(F.relu(self.conv3_4(self.padding(out))))

        out = F.relu(self.conv2_1(self.padding(out)))
        out = self.UpSample(F.relu(self.conv2_2(self.padding(out))))

        out = F.relu(self.conv1_1(self.padding(out)))
        out = self.conv1_2(self.padding(out))
        return out

# A vgg19 Sequential which is used up to Relu 4.1. To note is that the
# first layer is a 3,3 convolution, different from a standard vgg19

class Encoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg19 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True), # First layer from which Style Loss is calculated
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(inplace=True), # Second layer from which Style Loss is calculated
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1), # Third layer from which Style Loss is calculated
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True), # This is Relu 4.1 The output layer of the encoder.
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True)
            ).to(device)
        
        # state_dict_ = torch.load(state_dict, map_location=torch.device('cuda:0'))
        # self.vgg19.load_state_dict(state_dict_)

        encoder_children = list(self.vgg19.children())
        self.EncoderList = nn.ModuleList([nn.Sequential(*encoder_children[:4]), # Up to Relu 1.1
                                          nn.Sequential(*encoder_children[4:11]), # Up to Relu 2.1
                                          nn.Sequential(*encoder_children[11:18]), # Up to Relu 3.1
                                          nn.Sequential(*encoder_children[18:31]), # Up to Relu 4.1, also the
                                          ])                                       # input for the decoder

    def forward(self, x, intermediates=False): # if training use intermediates = True, to get the output of
        states = []                            # all the encoder layers to calculate the style loss
        for i in range(len(self.EncoderList)):
            x = self.EncoderList[i](x)

            if intermediates:       # All intermediate states get saved in states
                states.append(x)
        if intermediates:
            return states
        return x

