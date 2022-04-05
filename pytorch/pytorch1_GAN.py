import argparse 
import os
import numpy as np
import math

# torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

# torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

# preprocess : 파일 호출 및 전처리
# utils : 필요한 함수를 별도로 저장
# model : 모델은 class 단위로 설계
# train : 모델을 호출하고, 전처리한 데이터셋을 사용하여
# 그 외에도 이런 파일들이 있습니다.

# config.yaml : 하이퍼파라미터 저장
# solver : train을 포함한 여러 과정을 한번에 진행하게 만든 pipeline
# inference : 예측 및 생성 모델에서는 별도로 존재하는 경우가 있음

### Hyperparameter 설정 ###
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if torch.cuda.is_available() else False

# config.yaml 등의 파일에 저장하여 해당 파일을 불러오는 방법
# cmd에서 argument로 받아오는 방법
# argparse 라이브러리
# sys 라이브러리
# 취향에 따라 다를 수 있지만, 전체적인 custom이나 실험을 위해서는 argparse가 가장 나아보입니다. 
# 해당 argument의 default값과 자료형을 지정할 수 있고, 설명을 추가할 수 있다는 장점이 있습니다

# 여기 argument를 보면 다음 내용이 필요한 것을 알 수 있습니다.

# n_epochs : epoch 수
# batch_size : batch 크기
# lr, b1, b2 : adam optimzer의 파라미터
# n_cpu : cpu threads 수
# img_size : 이미지 사이즈
# channel : 이미지 채널, Mnist의 경우에는 흑백이미지이므로 1 channel로 사용하기 위함
# sample_interval : 제대로 훈련되고 있는지 sample 체크 간격 (출력)


### Generator ###
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img
    
# 각 모델은 nn.Module을 상속받습니다.
# 조금 신기하게 python unpacking을 사용하여 코드를 깔끔하게 만들었습니다.
# 사용한 layer는 다음과 같습니다. 여기서 사용하는 layer는 torch.nn에 정의되어 있습니다.
# Linear
# BatchNorm1d : 조금 신기한 것은 BatchNorm1d에 eps 파라미터에 0.8을 넣었다는 점..?
# LeackyReLU
# Tanh
# 구현에 따라 BatchNorm1d가 없애거나, Dropout층을 추가할 수 있습니다.
# 그리고 StarGAN 등의 구현체를 보면, init에 layer를 모두 nn.Sequential로 쌓고 forward 부분을 간소화합니다.
# torch에서 size()는 numpy의 shape과 똑같습니다. 여기서 size(0)은 batch_size를 의미합니다.

### Discriminator ###
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
    
### Dataset (DataLoader) ###
# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

### Loss Function & Optimizer ###
# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 손실함수는 Binary Cross Entropy 즉, BCELoss 를 사용합니다.
# 옵티마이저는 adam을 사용합니다.

### Training ###
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
os.makedirs("images", exist_ok=True)

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

# 훈련과정
# epochs * data 만큼 반복문을 돌립니다. data는 batch단위로 iter가 돌아갑니다.
# valid는 진짜(real)을 의미하는 1, fake는 가짜를 의미하는 0을 의미합니다.
# ramdom sampling한 tensor인 z를 Generator를 이용하여 이미지를 생성합니다.
# 생성된 이미지를 Discriminator에 넣어 참/거짓을 분별하고, 이게 얼만나 참인지를 loss로 사용합니다.
# 실제 이미지와 생성된 이미지를 넣고, 실제 이미지는 1, 생성 이미지는 0으로 구분하도록 계산합니다.
# 그리고 위의 각각 loss를 합친 것을 discriminator의 loss로 사용합니다.
# 각 back propagation은 zero_grad(), backward, step 과정으로 이루어지는데 이에 대한 설명은 생략합니다.