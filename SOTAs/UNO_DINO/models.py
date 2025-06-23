import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models import ResNet

class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes):
        super().__init__()

        self.prototypes = nn.Linear(output_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    def forward(self, x):
        return self.prototypes(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=1):
        super().__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MultiHead(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_prototypes, num_heads, num_hidden_layers=1
    ):
        super().__init__()
        self.num_heads = num_heads # 4

        # projectors
        self.projectors = torch.nn.ModuleList(
            [MLP(input_dim, hidden_dim, output_dim, num_hidden_layers) for _ in range(num_heads)]
        )

        # prototypes
        self.prototypes = torch.nn.ModuleList(
            [Prototypes(output_dim, num_prototypes) for _ in range(num_heads)] 
        )
        self.normalize_prototypes()

    @torch.no_grad()
    def normalize_prototypes(self):
        for p in self.prototypes:
            p.normalize_prototypes()

    def forward_head(self, head_idx, feats):
        z = self.projectors[head_idx](feats)
        z = F.normalize(z, dim=1)
        return self.prototypes[head_idx](z), z

    def forward(self, feats):
        out = [self.forward_head(h, feats) for h in range(self.num_heads)]
        return [torch.stack(o) for o in map(list, zip(*out))]


class MultiHeadModel(nn.Module):
    def __init__(
        self,
        args,
        arch,
        low_res,
        num_labeled,
        num_unlabeled,
        hidden_dim=2048,
        proj_dim=256,
        overcluster_factor=3, # discovery 10
        num_heads=5, # discovery 4
        num_hidden_layers=1,
        use_bn = False, # HACK for DINOHEAD
        nlayers = 3, # HACK for DINOHEAD
    ):
        super().__init__()

        self.args = args
        # backbone
        if args.arch == 'resnet18':
            self.encoder = models.__dict__[arch]()
            if args.use_pretrained_arch: # use pretrained backbone(ResNet18 on ImageNet with unsupervised pretraining)
                # load checkpoint
                model_save_dir = args.aim_root_path + 'trained_models/DINO/dino_resnet18_imagenet/checkpoint.pth' # final trained model
                checkpoint = torch.load(model_save_dir, map_location=lambda storage, loc: storage.cuda(args.gpu))
                state_dict = checkpoint['teacher']
                for k in list(state_dict.keys()):
                    if k.startswith('module.head'): 
                        del state_dict[k]
                self.encoder.load_state_dict(state_dict, strict=False)
                print('load pretrained ResNet18 weights successfully!')

            self.feat_dim = self.encoder.fc.weight.shape[1]
            self.encoder.fc = nn.Identity()
            # modify the encoder for lower resolution
            if low_res:
                self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.encoder.maxpool = nn.Identity()
                self._reinit_all_layers()

        elif args.arch == 'resnet50':
            if args.use_pretrained_arch: # use pretrained backbone(ResNet50 on ImageNet with unsupervised pretraining)
                self.encoder = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50') # (fc): Identity()
                # self.encoder.fc is nn.Identity()
                self.feat_dim = 2048
            else: # use unpretrained backbone
                self.encoder = models.__dict__[arch]() 
                # self.encoder.fc: (fc): Linear(in_features=2048, out_features=1000, bias=True)
                # self.encoder.fc.weight.shape  torch.Size([1000, 2048])
                self.feat_dim = self.encoder.fc.weight.shape[1]
            self.encoder.fc = nn.Identity()
            # modify the encoder for lower resolution
            if low_res:
                self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.encoder.maxpool = nn.Identity()
                self._reinit_all_layers()
        
        elif args.arch == 'DINO':
            self.encoder = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
            self.feat_dim = proj_dim
            # ----------------------
            # HOW MUCH OF BASE MODEL TO FINETUNE
            # ----------------------
            for m in self.encoder.parameters():
                m.requires_grad = False
            # Only finetune layers from block 'args.grad_from_block' onwards
            for name, m in self.encoder.named_parameters():
                if 'block' in name:
                    block_num = int(name.split('.')[1])
                    if block_num >= args.grad_from_block:
                        m.requires_grad = True
            # DINO head 
            layers = [nn.Linear(args.feat_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, proj_dim))
            self.mlp = nn.Sequential(*layers)
            
        else:
            raise NotImplementedError('model not supported {}'.format(args.arch))
        
        # heads
        self.head_lab = Prototypes(self.feat_dim, num_labeled) # Prototypes: no bias linear 256 --> num_labeled
        if num_heads is not None: # pretrain stage: False  discover stage: True  num_heads = None
            self.head_unlab = MultiHead(
                input_dim=self.feat_dim,
                hidden_dim=hidden_dim,
                output_dim=proj_dim,
                num_prototypes=num_unlabeled,
                num_heads=num_heads, # 4
                num_hidden_layers=num_hidden_layers,
            )
            self.head_unlab_over = MultiHead(
                input_dim=self.feat_dim,
                hidden_dim=hidden_dim,
                output_dim=proj_dim,
                num_prototypes=num_unlabeled * overcluster_factor, # discovery: num_unlabeled * 10
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )
        self.register_buffer("loss_per_head", torch.zeros(args.num_heads))

    @torch.no_grad()
    def _reinit_all_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def normalize_prototypes(self):
        self.head_lab.normalize_prototypes()
        if getattr(self, "head_unlab", False):
            self.head_unlab.normalize_prototypes()
            self.head_unlab_over.normalize_prototypes()

    def forward_heads(self, feats):
        out = {"logits_lab": self.head_lab(F.normalize(feats))}
        if hasattr(self, "head_unlab"):
            logits_unlab, proj_feats_unlab = self.head_unlab(feats)
            logits_unlab_over, proj_feats_unlab_over = self.head_unlab_over(feats)
            out.update(
                {
                    "logits_unlab": logits_unlab,
                    "proj_feats_unlab": proj_feats_unlab,
                    "logits_unlab_over": logits_unlab_over,
                    "proj_feats_unlab_over": proj_feats_unlab_over,
                }
            )
        return out

    def forward(self, views):
        if isinstance(views, list): # train stage   discover: view list 4 [bs,3,32,32][bs,3,32,32][bs,3,18,18][bs,3,18,18]
            if self.args.arch in ['resnet18', 'resnet50']:
                feats = [self.encoder(view) for view in views] # list 4 [bs,512][bs,512][bs,512][bs,512]
            elif self.args.arch == 'DINO':
                feats = [self.mlp(self.encoder(view)) for view in views]
            else:
                raise NotImplementedError('model not supported {}'.format(self.args.arch))
            out = [self.forward_heads(f) for f in feats] # list 4 out[0]: (['logits_lab', 'logits_unlab', 'proj_feats_unlab', 'logits_unlab_over', 'proj_feats_unlab_over'])
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        else:   # test stage
            if self.args.arch in ['resnet18', 'resnet50']:
                feats = self.encoder(views)
            elif self.args.arch == 'DINO':
                feats = self.mlp(self.encoder(views))
            else:
                raise NotImplementedError('model not supported {}'.format(self.args.arch))
            out = self.forward_heads(feats)
            out["feats"] = feats
            return out

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
        x = torch.flatten(x, 1) # 512
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