#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:46:37 2024

@author: ajnas
"""

#imports

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



#congifuration

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
noise_dim = 64

#optimizer parameters
lr = 0.002
beta1 = 0.5
beta2 = 0.99

#training variables
epochs = 20

## Load MNIST datasets 

from torchvision import datasets, transforms as T

train_aug = T.Compose([
    T.RandomRotation((-20,+20)),
    T.ToTensor()    
    ])

trainset = datasets.MNIST('MNIST/', download=True, train = True, transform = train_aug)


#plot images

image, label = trainset[5]

plt.imshow(image.squeeze(), cmap='gray')

##Load the data into batches

from torch.utils.data import DataLoader
from torchvision.utils import make_grid

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

dataiter = iter(trainloader)

images, labels = next(dataiter)
print(images.shape)

#show the tensor images


def show_tensor_images(tensor_img, num_img=16, size=(1, 28, 28)):
    unflat_img = tensor_img.detach().cpu()
    img_grid = make_grid(unflat_img[:num_img], nrow=4)
    plt.imshow(img_grid.permute(1, 2, 0).squeeze())
    plt.show()
    

show_tensor_images(images,num_img=20)


#Discriminator network
"""
Simple binary classifier to tell whether the generated image belongs to real or fake datasets
"""
from torch import nn
from torchsummary import summary

#discimator block

def get_desc_block(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2)
        )


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        
        self.block_1 = get_desc_block(1, 16, (3,3), 2)
        self.block_2 = get_desc_block(16, 32,(5,5), 2)
        self.block_3 = get_desc_block(32, 64,(5,5), 2)
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=64, out_features=1)
        
    def forward(self,images):
        
        x1 = self.block_1(images)
        x2 = self.block_2(x1)
        x3 = self.block_3(x2)
        
        x4 = self.flatten(x3)
        x5 = self.linear(x4)
        return x5
    
    
#See the summary

D = Discriminator()
D.to(device)

summary(D, input_size=(1,28,28)) 


# Cretae generator
"""
input size (batch_size, nosie dimention)

need to reshape 
"""

def get_gen_block(in_channels, out_channels, kernel_size, stride, final_block=False):
    if final_block==True:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
            nn.Tanh()
            )
    
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
        )


class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        
        self.noise_dim = noise_dim
        self.block_1 = get_gen_block(noise_dim, 256, (3,3), 2)
        self.block_2 = get_gen_block(256, 128, (4,4), 1)
        self.block_3 = get_gen_block(128, 64, (3,3), 2)
        
        self.block_4 = get_gen_block(64, 1, (4,4), 2, final_block=True)
        
    def forward(self, r_noice_vec):
        x = r_noice_vec.view(-1, self.noise_dim,1,1)
        x1 = self.block_1(x)
        x2 = self.block_2(x1)
        x3 = self.block_3(x2)
        x4 = self.block_4(x3)
        
        return x4
    
G = Generator(noise_dim)
G.to(device)

summary(G, input_size=(1,noise_dim))
        
        
        
#replace random initialize weight to normal weights

def weight_ini(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight,0.0,0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight,0.0,0.02)
        nn.init.constant(m.bias,0)
        

D = D.apply(weight_ini)
G = G.apply(weight_ini)


#cretae loss functions

#real loss
def real_loss(desc_pred):
    criterion = nn.BCEWithLogitsLoss()
    ground_truth = torch.ones_like(desc_pred)
    loss = criterion(desc_pred, ground_truth)
    return loss

def fake_loss(desc_pred):
    criterion = nn.BCEWithLogitsLoss()
    ground_truth = torch.zeros_like(desc_pred)
    loss = criterion(desc_pred, ground_truth)
    return loss

#oprtimizer

D_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))
G_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))


#Training loop

for i in range(epochs):
    total_d_loss = 0.0
    total_g_loss = 0.0
    
    for real_img, _ in tqdm(trainloader):
        real_img = real_img.to(device)
        noise = torch.randn(batch_size, noise_dim,device=device)
        
        #find loss ad update tje weight of D
        
        D_opt.zero_grad()
        
        fake_img = G(noise)
        D_pred = D(fake_img)
        D_fake_loss = fake_loss(D_pred)
        
        
        D_pred = D(real_img)
        D_real_loss = real_loss(D_pred)
        
        D_loss = (D_fake_loss+D_real_loss)/2
        
        total_d_loss +=D_loss.item()
        
        D_loss.backward()
        D_opt.step()
        
        #find loss ad update the weight of G
        
        G_opt.zero_grad()
        
        noise = torch.randn(batch_size, noise_dim, device=device)
        
        fake_img = G(noise)
        D_pred = D(fake_img)
        G_loss = real_loss(D_pred) # we want dicriminator close to real value
        
        total_g_loss+=G_loss.item()
        
        G_loss.backward()
        G_opt.step()
    
    avg_d_loss = total_d_loss/len(trainloader)
    avg_g_loss = total_g_loss/len(trainloader)
    
    
    print("Epoch: {} | D_loss:{} | G_loss: {}" .format(i+1, avg_d_loss, avg_g_loss))
    
    
    show_tensor_images(fake_img)


