import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock as ResNetBasicBlock
from torchvision.models.resnet import Bottleneck as ResNetBottleneck

from torchvision.models import ResNet

class SelfsupervisedResNet(nn.Module):
    def __init__(self, args, block, num_blocks, num_classes=4, is_adapters = None):  # block = BasicBlock num_blocks = [2,2,2,2] num_classes=4
        super(SelfsupervisedResNet, self).__init__()
        self.args = args
        self.is_adapters = is_adapters
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # def the mixstyle module
        self.mixstyle = MixStyle(rate = self.args.changestlye_rate)
        print('Changestlye_rate: ', self.args.changestlye_rate)
        
        if args.resizecrop_size == 32: # *1
            self.linear = nn.Linear(512*block.expansion, num_classes) # for cifar10size 32 * 32
        elif args.resizecrop_size == 64: # *2
            self.linear = nn.Linear(512*4, num_classes) # for domainnet 64 * 64
        elif args.resizecrop_size == 128: # *4
            self.linear = nn.Linear(512*16, num_classes) # for domainnet 128 * 128
        elif args.resizecrop_size == 224: # *7
            self.linear = nn.Linear(512*49, num_classes) # for domainnet 224 * 224
        
        if is_adapters:
            self.parallel_conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1) # stride + 1
        layers = []
        for stride in strides:
            layers.append(block(self.args, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers) 

    def forward(self, x, domain_flag):
        x = self.mixstyle(x, domain_flag)
        if self.is_adapters:
            out = F.relu(self.bn1(self.conv1(x)+self.parallel_conv1(x)))
        else:
            out = F.relu(self.bn1(self.conv1(x))) # torch.Size([2*4*bs, 3, 128, 128]) --> ([2*4*bs, 64, 128, 128])
        out = self.mixstyle(out, domain_flag)
        out = self.layer1(out) # torch.Size([2*4*bs, 64, 128, 128]) --> ([2*4*bs, 64, 128, 128]) 32*32 --> 32*32
        out = self.layer2(out) # torch.Size([2*4*bs, 64, 128, 128]) --> ([2*4*bs, 128, 64, 64]) 32*32 --> 16*16
        out = self.layer3(out) # torch.Size([2*4*bs, 128, 64, 64]) --> ([2*4*bs, 256, 32, 32]) 16*16 --> 8*8
        out = self.layer4(out) # torch.Size([2*4*bs, 256, 32, 32]) --> ([2*4*bs, 512, 16, 16]) 8*8 --> 4*4
        out = F.avg_pool2d(out, 4) # torch.Size([2*4*bs, 512, 16, 16]) --> ([2*4*bs, 512, 4, 4]) 4*4--> 1*1
        out = out.view(out.size(0), -1) # torch.Size([2*4*bs, 8192]) 512
        out = self.linear(out) # torch.Size([2*4*bs, 8192]) --> ([2*4*bs, 4])
        return out

class BasicBlock(nn.Module):
    expansion = 1  # for cifar10 32 * 32

    def __init__(self, args, in_planes, planes, stride=1): 
        # in_planes = [64,64],[64,128],[128,256],[256,512] 
        # planes = [64,64],[128,128],[256,256],[512,512]
        # stride = [1,1],[2,1],[2,1],[2,1]
        super(BasicBlock, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        self.is_padding = 0 
        if stride != 1 or in_planes != self.expansion*planes: # [, ] [True, ] [True, ] [True, ] --> [False, False] [True, False] [True, False] [True, False]
            self.shortcut = nn.AvgPool2d(2)
            if in_planes != self.expansion*planes:
                self.is_padding = 1

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.is_padding:
            shortcut = self.shortcut(x)
            out += torch.cat([shortcut,torch.zeros(shortcut.shape).float().cuda(self.args.gpu, non_blocking=True)],1)
        else:
            out += self.shortcut(x)
        out = F.relu(out)
        return out

class SupervisedResNet(nn.Module):
    def __init__(self, args, block, num_blocks, num_labeled_classes=5, num_unlabeled_classes=5):
        super(SupervisedResNet, self).__init__()
        self.args = args
        self.in_planes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2   = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3   = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4   = self._make_layer(block, 512, num_blocks[3], stride=2)
        # def the mixstyle module
        self.mixstyle = MixStyle(rate = self.args.changestlye_rate)
        print('Changestlye_rate: ', self.args.changestlye_rate)

        if args.resizecrop_size == 32: # *1
            self.head1 = nn.Linear(512*block.expansion, num_labeled_classes)
            self.head2 = nn.Linear(512*block.expansion, num_unlabeled_classes)
            self.mlp = nn.Sequential(nn.Linear(512*block.expansion, args.proj_dim))
        elif args.resizecrop_size == 64: # *2
            self.head1 = nn.Linear(512*4, num_labeled_classes)
            self.head2 = nn.Linear(512*4, num_unlabeled_classes)
            self.mlp = nn.Sequential(nn.Linear(512*4, args.proj_dim))
        elif args.resizecrop_size == 128: # *4
            self.head1 = nn.Linear(512*16, num_labeled_classes)
            self.head2 = nn.Linear(512*16, num_unlabeled_classes)
            self.mlp = nn.Sequential(nn.Linear(512*16, args.proj_dim))
        elif args.resizecrop_size == 224: # *7
            self.head1 = nn.Linear(512*49, num_labeled_classes)
            self.head2 = nn.Linear(512*49, num_unlabeled_classes)
            self.mlp = nn.Sequential(nn.Linear(512*49, args.proj_dim))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.args, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, domain_flag):
        x = self.mixstyle(x, domain_flag)
        out = F.relu(self.bn1(self.conv1(x))) # 128 3 32 32 --> 128 64 32 32
        out = self.mixstyle(out, domain_flag)
        out = self.layer1(out)     
        out = self.layer2(out)     
        out = self.layer3(out)     
        out = self.layer4(out)    
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        feature = self.mlp(out)
        out = F.relu(out)
        feature = F.relu(feature)
        out1 = self.head1(out)
        out2 = self.head2(out)
        return out1, out2, feature

def style_encoder(args):
    if args.style_arch == 'resnet18':
        model = ResNet_mix(baseblock = ResNetBasicBlock, layers = [2,2,2,2], args = args)
    elif args.style_arch == 'resnet50':
        model = ResNet_mix(baseblock = ResNetBottleneck, layers = [3,4,6,3], args = args)
    else:
        model = None
        raise NotImplementedError('model not supported {}'.format(args.style_arch))
    return model

class ResNet_mix(ResNet): # for supervised learning
    def __init__(self, baseblock, layers, args):
        super(ResNet_mix, self).__init__(block = baseblock, layers = layers)
        self.mixstyle = MixStyle(rate = args.changestlye_rate)
        print('Changestlye_rate: ', args.changestlye_rate)
        layer = []
        layer.append(nn.GELU())
        layer.append(nn.Linear(256, 128))
        layer.append(nn.GELU())
        layer.append(nn.Linear(128, 10))
        
        self.mlp = nn.Sequential(nn.Linear(512, args.proj_dim))

    def forward(self, x, domain_flag):
        # base encoder
        x = self.mixstyle(x, domain_flag) # torch.Size([4*bs, 3, 128, 128]) torch.Size([2*bs])
        x = self.conv1(x) # update channel 3 --> 64
        x = self.bn1(x) # keep
        x = self.relu(x) # keep
        x = self.maxpool(x) #  1/2 h*w

        x = self.mixstyle(x, domain_flag)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        style_feature = self.mlp(x)
        return style_feature

class MixStyle(nn.Module):
    # transfer target style to source
    def __init__(self, rate = 0.5):
        super().__init__()
        self.eps = 1e-6
        self.rate = rate

    def __repr__(self):
        return f'ChangeStyle(rate = {self.rate})'
    
    def match_input(self, x, domain_flag):
        batch_size = len(domain_flag)
        num_target = int(sum(domain_flag))
        num_source = batch_size - num_target
        if num_source == 0: # only target domain data inputs
            source_sample = None
            matched_target = None
            target_sample = x
        elif num_target == 0: # only source domain data inputs
            source_sample = x
            matched_target = None
            target_sample = None
        else:
            source_sample = x[domain_flag == 0] # split the source data
            target_sample = x[domain_flag == 1] # split the target data

            idx = torch.randint(num_target,(num_source,)) # sampling idx data from the target
            matched_target = target_sample[idx] # match the target

        return source_sample, matched_target, target_sample
    
    def change_style(self, x): # 
        # accelerate the change style, for batch input(2*bs, 3, 128, 128) 2*bs: source and matched target
        batch_size = int(x.size(0) / 2)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig
        mu2 = torch.cat((mu[batch_size:], mu[:batch_size]), dim = 0)
        sig2 = torch.cat((sig[batch_size:], sig[:batch_size]), dim = 0)

        mu_mix = mu*(1-self.rate) + mu2 * self.rate # rate = 0 no changestyle, rate = 1 all change style
        sig_mix = sig*(1-self.rate) + sig2 * self.rate
        mixed = x_normed*sig_mix + mu_mix
        Mixed_source, _ = mixed.chunk(2)

        return Mixed_source

    def forward(self, x, domain_flag):
        if not self.rate: # rate = 0, then no change style
            return x
        else:
            source_sample, matched_target, target_sample = self.match_input(x, domain_flag)
            if source_sample is None: # source_sample is empty, all is target domain data
                return target_sample
            elif target_sample is None: # target_sample is empty, all is source domain data
                return source_sample
            else:
                Mixed_source_batch = self.change_style(torch.cat([source_sample, matched_target], dim = 0))
                Mixed = torch.cat([Mixed_source_batch, target_sample], dim = 0)
                return Mixed