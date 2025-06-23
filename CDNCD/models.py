import torch
import torch.nn as nn
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck

class DINOHead(nn.Module):
    def __init__(self, args, use_bn=False, norm_last_layer=True, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        in_dim = args.feat_dim
        out_dim =  args.mlp_out_dim # args.num_classes
        nlayers = args.num_mlp_layers
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0: # this
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, args.dim_style))  # style feature dimension
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        logits = self.last_layer(x)
        return x_proj, logits

def base_model(args):
    if args.style_arch == 'resnet18':
        model = ResNet_mix(baseblock = BasicBlock, layers = [2,2,2,2], args = args)
    elif args.style_arch == 'resnet50':
        model = ResNet_mix(baseblock = Bottleneck, layers = [3,4,6,3], args = args)
    else:
        model = None
        raise NotImplementedError('model not supported {}'.format(args.style_arch))
    return model

class ResNet_mix(ResNet): # for supervised learning
    def __init__(self, baseblock, layers, args):
        super(ResNet_mix, self).__init__(block = baseblock, layers = layers)
        self.mixstyle = MixStyle(rate = args.changestyle_rate)
        print('Changestyle_rate: ', args.changestyle_rate)
        layer = []
        layer.append(nn.GELU())
        layer.append(nn.Linear(256, 128))
        layer.append(nn.GELU())
        layer.append(nn.Linear(128, 10))
        
        self.mlp = nn.Sequential(nn.Linear(512, args.dim_style))

    def forward(self, x, domain_flag):
        # base encoder
        x = self.mixstyle(x, domain_flag)
        x = self.conv1(x) # update channel
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
        # domain_flag is a tensor of length batchsize, storing 0, 1. 0 representing source and 1 representing target.
        # Here, based on domain_flag, the number of sources and targets is matched: making the number of targets the same as that of sources. (Simply sampling directly from targets with the same number of source)
        # The output contains Z[source_sample, matched_target, target_sample]
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