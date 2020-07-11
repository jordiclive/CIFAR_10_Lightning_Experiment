import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn import Conv2d, MaxPool2d
from pytorch_lightning.callbacks import ModelCheckpoint


from data.data import CIFAR10
#device = 'cpu', don't need to specify device, Lightning deals with it
dtype = torch.float32




def mish(x):
    return (x * torch.tanh(F.softplus(x)))


class MISH(nn.Module):

    def __init__(self, inplace: bool = False):
        super(MISH, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return mish(input)


# define resnet building blocks

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()

        self.left = nn.Sequential(Conv2d(inchannel, outchannel, kernel_size=3,
                                         stride=stride, padding=1, bias=False),
                                  nn.BatchNorm2d(outchannel),
                                  MISH(),
                                  Conv2d(outchannel, outchannel, kernel_size=3,
                                         stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(outchannel))

        self.shortcut = nn.Sequential()

        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(Conv2d(inchannel, outchannel,
                                                 kernel_size=1, stride=stride,
                                                 padding=0, bias=False),
                                          nn.BatchNorm2d(outchannel))

    def forward(self, x):
        out = self.left(x)

        out += self.shortcut(x)

        out = mish(out)

        return out

    # define resnet


class ResNet(pl.LightningModule):

    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()

        self.inchannel = 64
        self.conv1 = nn.Sequential(Conv2d(3, 64, kernel_size=3, stride=1,
                                          padding=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())

        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.maxpool = MaxPool2d(4)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = []

        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))

            self.inchannel = channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.maxpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

# training, eval mode, optimizer step etc. dealt with by lightnigng
    def training_step(self,batch,batch_idx):
        x, y = batch
        
        x = x.to( dtype=dtype)  
        y = y.to( dtype=torch.long)

        scores = self(x)
        loss = F.cross_entropy(scores, y)

        # Zero out all of the gradients for the variables which the optimizer
        # will update.
       # optimizer.zero_grad()
        log = {'train_loss':loss}
        # This is the backwards pass: compute the gradient of the loss with
        # respect to each  parameter of the model.
        return {'loss':loss,'log':log} # loss file to use  , logs purely for logging, eg. tensorboard

        # Actually update the parameters of the model using the gradients

    def validation_step(self,batch,batch_idx):

        x, y = batch
        x = x.to(dtype=dtype) 
        y = y.to( dtype=torch.long)

        scores = self(x)
        val_loss = F.cross_entropy(scores, y)

        # Zero out all of the gradients for the variables which the optimizer
        # will update.
        # optimizer.zero_grad()

        # This is the backwards pass: compute the gradient of the loss with
        # respect to each  parameter of the model.
        return {'val_loss': val_loss}

    def validation_epoch_end(self,outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {'avg_val_loss': val_loss}
        self.logger.experiment.add_scalar('validation loss',
                            val_loss,
                            self.current_epoch)
        return {'log': log,'val_loss': val_loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())

def ResNet18():
    return ResNet(ResidualBlock)

if __name__ == "__main__":
    DATA = CIFAR10()
    train_loader = DATA.trainloader
    val_loader = DATA.testloader
    # Lightning has own way to store data loaders
    model = ResNet18()
    #checkpoint_callback = ModelCheckpoint(filepath='/content/gdrive/My Drive/CIFAR_10/CIFAR_10_colab_experiment/checkpoints/weights.ckpt',verbose=True,monitor='val_loss',mode='min')
    # checkpoint if want to start from
    trainer = pl.Trainer(max_epochs=15,gpus=1)#checkpoint_callback=checkpoint_callback,resume_from_checkpoint='/content/gdrive/My Drive/CIFAR_10/CIFAR_10_colab_experiment/checkpoints/_ckpt_epoch_4.ckpt')
    trainer.fit(model,train_dataloader=train_loader,val_dataloaders=val_loader)
	# can do fast-dev run etc.

