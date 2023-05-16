import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import os
import cv2
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.neighbors  import KNeighborsClassifier 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
from evaluation import *
import math
import csv

kernels=[3,3,3,3]
paddings=[1,1,1,1]
paddings_out=[1,1,1,1]

class EncoderBlock(nn.Module):
    def __init__(self,inChannel,OutChannels):
        super(EncoderBlock,self).__init__()
        modules=nn.ModuleList()
        for i in range(4):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(inChannel, OutChannels[i],
                                kernel_size= kernels[i], stride= 2, padding  = paddings[i]),
                        nn.BatchNorm2d(OutChannels[i]),
                        nn.LeakyReLU()
                )
            )
        self.ml=modules.to('cuda:0')

    def forward(self,x):
        result=[]
        for i in range(4):
            result.append(self.ml[i](x))
        
        
        return torch.cat(result,1)

class DecoderBlock(nn.Module):
    def __init__(self,inChannel,OutChannels):
        super(DecoderBlock,self).__init__()
        modules=nn.ModuleList()
        tot=0
        for i in range(4):
            modules.append(
                nn.Sequential(
                        nn.ConvTranspose2d(inChannel,
                                        OutChannels[i],
                                        kernel_size=kernels[i],
                                        stride = 2,
                                        padding= paddings[i],
                                        output_padding=paddings_out[i]),
                        nn.BatchNorm2d(OutChannels[i]),
                        nn.LeakyReLU()
            ))
            tot+=OutChannels[i]
        self.ml=modules.to('cuda:0')
        

    def forward(self,x):
        result=[]
        for i in range(4):
            result.append(self.ml[i](x))
        x=torch.cat(result,1)
        return x



class ImageVAE(nn.Module):

    def __init__(self,input_channel=3,feature_size=2048,img_size=(256,512)):
        super(ImageVAE,self).__init__()

        self.feature_size=feature_size
        
        modules = []
        self.hidden_dims = [input_channel,64,128,256,512,1024]

        # Build Encoder
        for i,h in enumerate(self.hidden_dims[:-1]):
            modules.append(nn.Sequential(
                    nn.Conv2d(h, self.hidden_dims[i+1],
                                kernel_size= 3, stride= 2, padding  = 1),
                        nn.BatchNorm2d(self.hidden_dims[i+1]),
                        nn.LeakyReLU()
                ))
        
        self.encoder=nn.Sequential(*modules).to('cuda:0')
        
        self.final_size=(img_size[0]/2**(len(self.hidden_dims)-1),img_size[0]/2**(len(self.hidden_dims)-1))

        self.fcMu = nn.Linear(self.hidden_dims[-1]*self.final_size[0]*self.final_size[1], feature_size).to('cuda:2')
        self.fcVar = nn.Linear(self.hidden_dims[-1]*self.final_size[0]*self.final_size[1], feature_size).to('cuda:3')

        modules = [[],[],[],[]]
        self.decoderInput = nn.Linear(feature_size, self.hidden_dims[-1]*self.final_size[0]*self.final_size[1]).to('cuda:0')

        self.hidden_dims.reverse()

        decoder=[]
        for i in range(len(self.hidden_dims) - 2):
            decoder.append(nn.Sequential(
                            nn.ConvTranspose2d(self.hidden_dims[i],
                                            self.hidden_dims[i+1],
                                            kernel_size=3,
                                            stride = 2,
                                            padding= 1,
                                            output_padding=1),
                            nn.BatchNorm2d(self.hidden_dims[i+1]),
                            nn.LeakyReLU()
                ))
        
        self.decoder=nn.Sequential(*decoder).to('cuda:0')

        self.finalLayer = nn.Sequential(
                            nn.ConvTranspose2d(self.hidden_dims[i+1],
                                            input_channel,
                                            kernel_size=7,
                                            stride=2,
                                            padding=3,
                                            output_padding=1),
                            nn.BatchNorm2d(input_channel),
                            nn.LeakyReLU()
                            ).to('cuda:0')

    
    def forward(self,input,encode=False):

        result = self.encoder(input).to('cuda:2')
        
        result = torch.flatten(result, start_dim=1)

        mu = self.fcMu(result)
        logVar = self.fcVar(result.to('cuda:3'))

        std = torch.exp(0.005 * logVar)

        decoderInput=((torch.randn_like(std) * std) + mu.to('cuda:3')).to('cuda:0')
        
        if not encode:
            result=self.decoderInput(decoderInput).to('cuda:0')
            result = result.view(-1, self.hidden_dims[0], self.final_size[0], self.final_size[1])
            result = self.decoder(result)
            result=self.finalLayer(result)
        
        else:
            result=None

        return decoderInput,result