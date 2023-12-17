from __future__ import absolute_import, print_function
import random
import os
import sys
import time
import yaml
import torch
from PIL import Image
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
from torchvision.transforms import InterpolationMode
from reid.utils.data import transforms as T
from .utils.meters import AverageMeter
import torchvision.transforms as transforms

import numpy as np
import re
import os.path as osp

from adain_model.adain_random import StyleTransferNetwork
        
def calc_mean_std(input, eps=1e-5):
    batch_size, channels = input.shape[:2]

    reshaped = input.view(batch_size, channels, -1) # Reshape channel wise
    mean = torch.mean(reshaped, dim = 2).view(batch_size, channels, 1, 1) # Calculat mean and reshape
    std = torch.sqrt(torch.var(reshaped, dim=2)+eps).view(batch_size, channels, 1, 1) # Calculate variance, add epsilon (avoid 0 division),
                                                                                    # calculate std and reshape
    return mean, std

def Random_Mix_Style(x, s):
    x_org = x
    B, C, W, H = x.size(0), x.size(1), x.size(2), x.size(3)
    mean_x, std_x = calc_mean_std(x)
    stye_mean_x, stye_std_x = calc_mean_std(s)
    lmda = torch.empty((B,C, W, H)).uniform_(0.1,0.3).cuda(device='cuda:0')

    new_x = lmda * (stye_std_x * ((x - mean_x) / std_x) + stye_mean_x) + (1 - lmda) * x_org
    return new_x

def Mix_Style(x):
    x_org = x
    B, C, W, H = x.size(0), x.size(1), x.size(2), x.size(3)
    mean_x, std_x = calc_mean_std(x)
    stye_mean_x, stye_std_x = calc_mean_std(x[torch.randperm(B)])
    lmda = torch.empty((B,C, W, H)).uniform_(0.1,0.3).cuda(device='cuda:0')

    new_x = lmda * (stye_std_x * ((x - mean_x) / std_x) + stye_mean_x) + (1 - lmda) * x_org
    return new_x

def EDM_Mix_Style(x):
    B, C, W, H = x.size(0), x.size(1), x.size(2), x.size(3)
    x_view = x.view(B, C, -1)
    value_x, index_x = torch.sort(x_view)  # sort inputs
    lmda = torch.empty((B,C,1)).uniform_(0.1,0.3).cuda(device='cuda:0')
    lmda = lmda.to(x.device)

    inverse_index = index_x.argsort(-1)
    x_view_copy = value_x[torch.randperm(B)].gather(-1, inverse_index)
    new_x = x_view + (x_view_copy - x_view.detach()) * lmda
    return new_x.view(B, C, W, H)

def DSU(x, eps=1e-6):
    x_org = x
    B, C, W, H = x.size(0), x.size(1), x.size(2), x.size(3)
    mean = x.mean(dim=[2, 3], keepdim=False)
    std = (x.var(dim=[2, 3], keepdim=False) + eps).sqrt()
    
    sqrtvar_mu = ((mean.var(dim=0, keepdim=True) + eps).sqrt()).repeat(mean.shape[0], 1)
    sqrtvar_std = ((std.var(dim=0, keepdim=True) + eps).sqrt()).repeat(std.shape[0], 1)

    beta = mean + (torch.randn_like(sqrtvar_mu)) * sqrtvar_mu 
    gamma = std + (torch.randn_like(sqrtvar_std)) * sqrtvar_std

    x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
    x_style = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

    lmda = torch.empty((B,C,W,H)).uniform_(0.1,0.3).cuda(device='cuda:0')
    new_x = lmda * x_style + (1 - lmda) * x_org

    return new_x

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def recover(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = inp * 255.0
    inp = np.clip(inp, 0, 255)
    return inp

toPIL = transforms.ToPILImage(mode="RGB")
class BaseTrainer(object):
    def __init__(self, model, criterion, args):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.args = args
        self.clip_value = self.args.clip_value
        
        # The freezed adain model
        self.adain = StyleTransferNetwork(device=torch.device("cuda:0")) 
        state_dict_encoder = torch.load(osp.join(args.adain_model_path, "QAConv_dg_ada_v2_learn_v1_new/adain_model/vgg_normalised.pth"))
        state_dict_decoder = torch.load(osp.join(args.adain_model_path, "QAConv_dg_ada_v2_learn_v1_new/adain_model/StyleTransfer Checkpoint Iter_ 120000.tar"))
        self.adain.encoder.vgg19.load_state_dict(state_dict_encoder, strict=False)
        self.adain.decoder.load_state_dict(state_dict_decoder["Decoder"], strict=False)
        for k, v in self.adain.encoder.named_parameters():
            v.requires_grad = False
        for k, v in self.adain.decoder.named_parameters():
            v.requires_grad = False

    def recover_image(self, input):
        tensor = input.clamp(min=-1, max=1)
        low, high = float(tensor.min()), float(tensor.max())
        tensor.clamp_(min=low, max=high)
        out = tensor.sub_(low).div_(max(high - low, 1e-5))

        return out
        
    def create_distance_covariance(self, D1, D2):
        b, c , _, _ = D1.shape
        D1 = D1.view(b,c)
        D2 = D2.view(b,c)
        distance = torch.sqrt(torch.sum(D1*D2) / (b * b)) / torch.sqrt(torch.sqrt(torch.sum(D1*D1) / (b * b)) * torch.sqrt(torch.sum(D2*D2) / (b * b)))

        return distance

    # the style correlation loss
    def generate_fusion_dis_loss(self, mean_list, std_list):
        mean_dis_loss = (self.create_distance_covariance(mean_list[0], mean_list[1]) + self.create_distance_covariance(mean_list[0], mean_list[2]) + self.create_distance_covariance(mean_list[0], mean_list[3]) 
                       + self.create_distance_covariance(mean_list[1], mean_list[2]) + self.create_distance_covariance(mean_list[1], mean_list[3]) + self.create_distance_covariance(mean_list[2], mean_list[3])) / 6 
        
        std_dis_loss = (self.create_distance_covariance(std_list[0], std_list[1]) + self.create_distance_covariance(std_list[0], std_list[2]) + self.create_distance_covariance(std_list[0], std_list[3]) 
                       + self.create_distance_covariance(std_list[1], std_list[2]) + self.create_distance_covariance(std_list[1], std_list[3]) + self.create_distance_covariance(std_list[2], std_list[3])) / 6 
        
        dis_loss = (mean_dis_loss + std_dis_loss) / 2
        return dis_loss
    
    def generate_mean_std(self, style_img, content_qiangdu_mean, content_qiangdu_std, content_mean, content_std):
        # the mean and standard deviation of a randomly selected image in the source domain
        style_img_content_feature = self.adain.encoder(style_img, False)
        style_mean_img, style_std_img = calc_mean_std(style_img_content_feature)
        # initialize the learned parameter in ASS module
        mean_s, std_s = nn.init.normal_(torch.zeros_like(content_mean), 0, 0.1), nn.init.normal_(torch.zeros_like(content_std), 1, 0.1)
        mean_s.requires_grad_(True)
        std_s.requires_grad_(True)
        # calculate the style (mean and std) for one style branch
        style_mean_final = (mean_s / torch.norm(mean_s, p=2,dim=1,keepdim=True) * torch.norm(content_qiangdu_mean, p=2,dim=1,keepdim=True)) * torch.randn(mean_s.size()).cuda(device='cuda:0') + style_mean_img
        style_std_final = (std_s / torch.norm(std_s, p=2,dim=1,keepdim=True) * torch.norm(content_qiangdu_std, p=2,dim=1,keepdim=True)) * torch.randn(std_s.size()).cuda(device='cuda:0') + style_std_img

        return style_mean_final, style_std_final, mean_s, std_s, style_mean_img, style_std_img

    def Adversarial_Style_Synthesis(self, img, style_img_0, style_img_1, style_img_2, targets, input):
        # we use the pre-trained Adain to caculate the style (feature mean and std) of content img.
        # the style feature transformation intensity (qiangdu) of content img is also calculated to make the generated style distribution consistent with content img.
        img_content_feature = self.adain.encoder(img, False)
        content_mean, content_std = calc_mean_std(img_content_feature)
        content_qiangdu_mean = (content_mean.var(0, keepdim=True) + 1e-6).sqrt()
        content_qiangdu_std = (content_std.var(0, keepdim=True) + 1e-6).sqrt()

        # the generated 3 different style branches. 
        # final: the final style for each branch. 
        # learn: the learned parameter for each branch. 
        # init: the style of random selected style img. 
        style_mean_final_0, style_std_final_0, style_mean_learn_0, style_std_learn_0, style_mean_img_0, style_std_img_0 = self.generate_mean_std(style_img_0, content_qiangdu_mean, content_qiangdu_std, content_mean, content_std)
        style_mean_final_1, style_std_final_1, style_mean_learn_1, style_std_learn_1, style_mean_img_1, style_std_img_1 = self.generate_mean_std(style_img_1, content_qiangdu_mean, content_qiangdu_std, content_mean, content_std)
        style_mean_final_2, style_std_final_2, style_mean_learn_2, style_std_learn_2, style_mean_img_2, style_std_img_2 = self.generate_mean_std(style_img_2, content_qiangdu_mean, content_qiangdu_std, content_mean, content_std)
        # generate the fusion style
        weight = torch.softmax(torch.randn(3, device=img.device), dim=0)
        style_mean_fusion = style_mean_final_0 * weight[0] + style_mean_final_1 * weight[1] + style_mean_final_2 * weight[2]
        style_std_fusion = style_std_final_0 * weight[0] + style_std_final_1 * weight[1] + style_std_final_2 * weight[2]
        # init the optimizer of ASS module, only train the learned parameter in the ASS module.
        learned_weight = [style_mean_learn_0, style_std_learn_0, style_mean_learn_1, style_std_learn_1, style_mean_learn_2, style_std_learn_2]
        optimizer_ASS = torch.optim.SGD(learned_weight, lr=self.args.lr_ASS, momentum=0.9, weight_decay=5e-4, nesterov=True)
        # generated the transferred Adain feature: feat_fusion
        feat_fusion = style_std_fusion * ((img_content_feature - content_mean) / (content_std)) + style_mean_fusion
        weight_matrix = torch.empty((feat_fusion.shape[0], feat_fusion.shape[1], feat_fusion.shape[2], feat_fusion.shape[3])).uniform_(self.args.omega_l, self.args.omega_H).cuda(img.device)
        # use the Adain decoder to generate the synthetic image
        input_ASS_list = []
        for j in range(img.shape[0]):
            img_ASS = self.adain.decoder(weight_matrix[j:j+1] * feat_fusion[j:j+1] + (1 - weight_matrix[j:j+1]) * img_content_feature[j:j+1])
            img_ASS = self.recover_image(img_ASS)
            input_ASS_list.append(img_ASS.cuda(img.device))
        input_ASS_tmp = torch.cat(input_ASS_list, dim=0)
        input_ASS_fusion = qaconv_train_transformers(input_ASS_tmp)
        # use the original image and the generated image (ASS) to calculate the loss_ASS and optimize the learned parameter in the ASS module
        with torch.cuda.amp.autocast():
            feat_org = self.model(input)
            loss_task_ASS, _, feat_ASS, _ = self._forward(input_ASS_fusion, targets)
            loss_sim = torch.mean((feat_ASS - feat_org).pow(2))  
            loss_adv = loss_task_ASS + self.args.varphi * loss_sim
            loss_cor = self.generate_fusion_dis_loss([content_mean, style_mean_final_0, style_mean_final_1, style_mean_final_2], [content_std, style_std_final_0, style_std_final_1, style_std_final_2])
            loss_ASS = -loss_adv + self.args.lambda_c * loss_cor
        optimizer_ASS.zero_grad()
        loss_ASS.backward()
        optimizer_ASS.step()
        # use the updated learned parameters to regain the fusion style 
        style_mean_final_0 = (style_mean_learn_0 / torch.norm(style_mean_learn_0, p=2,dim=1,keepdim=True) * torch.norm(content_qiangdu_mean, p=2,dim=1,keepdim=True)) * torch.randn(style_mean_learn_0.size()).cuda(device='cuda:0') + style_mean_img_0
        style_std_final_0 = (style_std_learn_0 / torch.norm(style_std_learn_0, p=2,dim=1,keepdim=True) * torch.norm(content_qiangdu_std, p=2,dim=1,keepdim=True)) * torch.randn(style_std_learn_0.size()).cuda(device='cuda:0') + style_std_img_0
        style_mean_final_1 = (style_mean_learn_1 / torch.norm(style_mean_learn_1, p=2,dim=1,keepdim=True) * torch.norm(content_qiangdu_mean, p=2,dim=1,keepdim=True)) * torch.randn(style_mean_learn_1.size()).cuda(device='cuda:0') + style_mean_img_1
        style_std_final_1 = (style_std_learn_1 / torch.norm(style_std_learn_1, p=2,dim=1,keepdim=True) * torch.norm(content_qiangdu_std, p=2,dim=1,keepdim=True)) * torch.randn(style_std_learn_1.size()).cuda(device='cuda:0') + style_std_img_1
        style_mean_final_2 = (style_mean_learn_2 / torch.norm(style_mean_learn_2, p=2,dim=1,keepdim=True) * torch.norm(content_qiangdu_mean, p=2,dim=1,keepdim=True)) * torch.randn(style_mean_learn_2.size()).cuda(device='cuda:0') + style_mean_img_2
        style_std_final_2 = (style_std_learn_2 / torch.norm(style_std_learn_2, p=2,dim=1,keepdim=True) * torch.norm(content_qiangdu_std, p=2,dim=1,keepdim=True)) * torch.randn(style_std_learn_2.size()).cuda(device='cuda:0') + style_std_img_2
        style_mean_fusion = style_mean_final_0 * weight[0] + style_mean_final_1 * weight[1] + style_mean_final_2 * weight[2]
        style_std_fusion = style_std_final_0 * weight[0] + style_std_final_1 * weight[1] + style_std_final_2 * weight[2]   
        # generate the final synthetic image for the next IFL module
        with torch.no_grad():
            input_ASS_list = []
            # generated the transferred Adain feature: feat_fusion
            feat_fusion = style_std_fusion * ((img_content_feature - content_mean) / (content_std)) + style_mean_fusion
            weight_matrix = torch.empty((feat_fusion.shape[0], feat_fusion.shape[1], feat_fusion.shape[2], feat_fusion.shape[3])).uniform_(self.args.omega_l, self.args.omega_H).cuda(img.device)
            # use the Adain decoder to generate the synthetic image
            for j in range(img.shape[0]):
                img_ASS_update = self.adain.decoder(weight_matrix[j:j+1] * feat_fusion[j:j+1] + (1 - weight_matrix[j:j+1]) * img_content_feature[j:j+1])
                img_ASS_update = self.recover_image(img_ASS_update)
                input_ASS_list.append(img_ASS_update.cuda(img.device))
            input_ASS = qaconv_train_transformers(torch.cat(input_ASS_list, dim=0))

        return input_ASS
        
    def train(self, epoch, data_loader, optimizer):
        # Creates once at the beginning of training
        scaler = torch.cuda.amp.GradScaler()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            self.model.eval()
            self.criterion.train()

            data_time.update(time.time() - end)
            # random select K (K = 3) images from the training samples as init style images
            input, img_orgs, style_img_0, style_img_1, style_img_2, targets = self._parse_data(inputs)
            # the Adversarial Style Synthesis module to generate novel image 
            input_ASS = self.Adversarial_Style_Synthesis(img_orgs, style_img_0, style_img_1, style_img_2, targets, input)
            
            optimizer.zero_grad()
            # Casts operations to mixed precision
            with torch.cuda.amp.autocast():
                loss, acc, feature, score = self._forward(input, targets)
                loss_ASS, acc_ASS, feature_ASS, score_ASS = self._forward(input_ASS.detach(), targets)
                
                if (loss or loss_ASS) is None:
                  continue  
                # Invariant Feature Learning module
                loss_sim = torch.mean((feature-feature_ASS).pow(2))
                loss = (loss + loss_ASS * self.args.lambda_t) / (1 + self.args.lambda_t) + loss_sim * self.args.varphi

            losses.update(loss.item(), targets.size(0))
            precisions.update(acc.item(), targets.size(0))

            if self.clip_value > 0:
                # Scales the loss, and calls backward() to create scaled gradients
                scaler.scale(loss).backward()
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            clip_grad_norm_(self.model.parameters(), self.clip_value)
            clip_grad_norm_(self.criterion.parameters(), self.clip_value)

            if self.clip_value > 0:
                # Unscales gradients and calls or skips optimizer.step()
                scaler.step(optimizer)
                # Updates the scale for next iteration
                scaler.update()
            else:
                optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            print('Epoch: [{}][{}/{}]. '
                  'Time: {:.3f} ({:.3f}). '
                  'Data: {:.3f} ({:.3f}). '
                  'Loss: {:.3f} ({:.3f}). '
                  'Prec: {:.2%} ({:.2%}).'
                  .format(epoch + 1, i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg,
                          losses.val, losses.avg,
                          precisions.val, precisions.avg), end='\r', file=sys.stdout.console)
            
        return losses.avg, precisions.avg

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError

qaconv_train_transformers = T.Compose([
        T.Pad(10),
        T.RandomCrop((384,128)),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(5), 
        T.ColorJitter(brightness=(0.5, 2.0), contrast=(0.5, 2.0), saturation=(0.5, 2.0), hue=(-0.1, 0.1)),
    ])

class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, img_orgs, style_img_0, style_img_1, style_img_2, fnames, pids, cams = inputs
        inputs = imgs.cuda()
        img_orgs = img_orgs.cuda()
        style_img_0 = style_img_0.cuda()
        style_img_1 = style_img_1.cuda()
        style_img_2 = style_img_2.cuda()
        targets = pids.cuda()
        return inputs, img_orgs, style_img_0, style_img_1, style_img_2, targets

    def _forward(self, inputs, targets):
        feature = self.model(inputs)
        loss, acc, score = self.criterion(feature, targets)
        finite_mask = loss.isfinite()
        if finite_mask.any():
            loss = loss[finite_mask].mean()
            acc = acc[finite_mask].mean()
        else:
            loss = acc = None
        return loss, acc, feature, score
