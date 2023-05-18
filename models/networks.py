from torchvision import models
import torch
import torch.nn as nn
import functools
import numpy as np
from torch.nn.functional import interpolate, pad

###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, norm='instance', gpu_ids=[], upsample_type='transconv', downsample_type='conv', input_size=(128, 256), n_attn_g=0, n_attn_l=0, proj_factor_g=4, heads_g=4, dim_head_g=128, proj_factor_l=4, heads_l=4, dim_head_l=128):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'global':
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global,
                               n_blocks_global, norm_layer, downsample_type=downsample_type, upsample_type=upsample_type,
                               input_size=input_size,
                               n_attn_g=n_attn_g, proj_factor_g=proj_factor_g, heads_g=heads_g, dim_head_g=dim_head_g)
    elif netG == 'local':
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
                             n_local_enhancers, n_blocks_local, norm_layer, downsample_type=downsample_type, upsample_type=upsample_type,
                             input_size=input_size,
                             n_attn_g=n_attn_g, proj_factor_g=proj_factor_g, heads_g=heads_g, dim_head_g=dim_head_g, n_attn_l=n_attn_l, proj_factor_l=proj_factor_l, heads_l=heads_l, dim_head_l=dim_head_l)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf,
                       n_downsample_global, norm_layer)
    else:
        raise('generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(
        input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def define_MR_D(ndf, n_layers_D, input_nc, norm='instance', use_sigmoid=False, num_D=1, gpu_ids=[], base_nfft=2048, window=None, min_value=1e-7, mdct_type='4', normalizer=None, getIntermFeat=False, abs_spectro=False):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiResolutionDiscriminator(ndf=ndf, n_layers=n_layers_D, input_nc=input_nc, norm_layer=norm_layer, num_D=num_D, base_nfft=base_nfft, window=window,
                                        min_value=min_value, mdct_type=mdct_type, use_sigmoid=use_sigmoid, normalizer=normalizer, getIntermFeat=getIntermFeat, abs_spectro=abs_spectro)
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 device='cuda'):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.device = device
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.shape != input.shape))
            if create_label:
                self.real_label_var = torch.full(size=input.size(),fill_value=self.real_label, device=self.device, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.shape != input.shape))
            if create_label:
                self.fake_label_var = torch.full(size=input.size(),fill_value=self.fake_label, device=self.device, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * \
                self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class SpecLoss(nn.Module):
    def __init__(self) -> None:
        super(SpecLoss, self).__init__()

    def forward(self, x, y):
        # input shape B,C,H,W
        N = x.shape[-1]
        spec_loss = torch.norm(x-y, p='fro', dim=(-1, -2)) / \
            torch.norm(x, p='fro', dim=(-1, -2))
        mag_loss = torch.norm(torch.log10(
            torch.abs(x)+1e-7) - torch.log10(torch.abs(y)+1e-7), p=1, dim=(-1, -2)) / N
        return torch.mean(spec_loss+mag_loss)
##############################################################################
# Generator
##############################################################################


class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect', downsample_type='conv', upsample_type='transconv', n_attn_g=0, n_attn_l=0, input_size=(128, 256), proj_factor_g=4, heads_g=4, dim_head_g=128, proj_factor_l=4, heads_l=4, dim_head_l=128):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer,
                                       downsample_type=downsample_type, upsample_type=upsample_type,
                                       input_size=tuple(map(lambda x: x//2, input_size)), n_attn_g=n_attn_g, proj_factor_g=proj_factor_g, heads_g=heads_g, dim_head_g=dim_head_g).model
        # get rid of final convolution layers
        model_global = [model_global[i] for i in range(len(model_global)-3)]
        self.model = nn.Sequential(*model_global)

        # downsample
        if downsample_type == 'conv':
            downsample_layer = nn.Conv2d
        elif downsample_type == 'resconv':
            downsample_layer = ConvResBlock
        else:
            raise NotImplementedError(
                'downsample layer [{:s}] is not found'.format(downsample_type))
        # upsample
        if upsample_type == 'transconv':
            upsample_layer = nn.ConvTranspose2d
        elif upsample_type == 'interpolate':
            upsample_layer = InterpolateUpsample
        else:
            raise NotImplementedError(
                'upsample layer [{:s}] is not found'.format(upsample_type))
        ###### local enhancer layers #####
            # downsample
        ngf_global = ngf * (2**(n_local_enhancers-1))
        model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                            norm_layer(ngf_global), nn.ReLU(True),
                            downsample_layer(ngf_global, ngf_global * 2,
                                             kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf_global * 2), nn.ReLU(True)]
        # residual blocks
        model_upsample = []
        for i in range(n_blocks_local):
            model_upsample += [ResnetBlock(ngf_global * 2,
                                           padding_type=padding_type, norm_layer=norm_layer)]
        # attention bottleneck
        if n_attn_l > 0:
            middle = n_blocks_local//2
            # 8x downsample
            down = [downsample_layer(ngf_global * 2, ngf_global,
                                     kernel_size=3, stride=2, padding=1),
                    norm_layer(ngf_global), nn.ReLU(True)]
            down += [downsample_layer(ngf_global, ngf_global,
                                      kernel_size=3, stride=2, padding=1),
                     norm_layer(ngf_global), nn.ReLU(True)]*2
            down = nn.Sequential(*down)
            model_upsample.insert(middle, down)

            middle += 1
            input_size = tuple(map(lambda x: x//16, input_size))
            from bottleneck_transformer_pytorch import BottleStack
            attn_block = BottleStack(dim=ngf_global, fmap_size=input_size, dim_out=ngf_global*2, num_layers=n_attn_l, proj_factor=proj_factor_l,
                                     downsample=False, heads=heads_l, dim_head=dim_head_l, activation=nn.ReLU(True), rel_pos_emb=False)
            model_upsample.insert(middle, attn_block)
            model_upsample += [upsample_layer(in_channels=ngf_global*2, out_channels=ngf_global*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                               norm_layer(ngf_global), nn.ReLU(True)]*3

        model_upsample += [upsample_layer(in_channels=ngf_global*2, out_channels=ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                           norm_layer(ngf_global), nn.ReLU(True)]

        # final convolution
        model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(
            ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

        self.model1_1 = nn.Sequential(*model_downsample)
        self.model1_2 = nn.Sequential(*model_upsample)

        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=[1, 1], count_include_pad=False)
        self.freeze = False

    def forward(self, input):
        # create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        # output at coarest level
        output_prev = self.model(input_downsampled[-1])
        # build up one layer at a time
        model_downsample = self.model1_1
        model_upsample = self.model1_2
        input_i = input_downsampled[0]
        output_prev = model_upsample(
                model_downsample(input_i) + output_prev)
        return output_prev

    def set_freeze(self, freeze_global_d=True, freeze_global_u=False, freeze_local_d=True, freeze_local_u=False):
        print("The following layers will be freezed:")
        '''Freeze downsample layers'''
        print('Global:')
        for name, layer in self.model.named_children():
            module_name = layer.__class__.__name__
            if 'Conv2d' in module_name or 'ConvResBlock' in module_name:
                if freeze_global_d:
                    print(name, module_name)
                for param in layer.parameters():
                    param.requires_grad = not freeze_global_d
            elif 'InterpolateUpsample' in module_name or 'ConvTranspose2d' in module_name or 'ResnetBlock' in module_name or 'BottleStack' in module_name:
                if freeze_global_u:
                    print(name, module_name)
                for param in layer.parameters():
                    param.requires_grad = not freeze_global_u
        print('Loacl:')
        for name, layer in self.model1_1.named_children():
            module_name = layer.__class__.__name__
            for param in layer.parameters():
                if freeze_local_d:
                    print(name, module_name)
                param.requires_grad = not freeze_local_d

        for name, layer in self.model1_2.named_children():
            module_name = layer.__class__.__name__
            for param in layer.parameters():
                if freeze_local_u:
                    print(name, module_name)
                param.requires_grad = not freeze_local_u


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', upsample_type='transconv', downsample_type='conv', n_attn_g=0, input_size=(128, 256), proj_factor_g=4, heads_g=4, dim_head_g=128):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(
            input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        # downsample
        if downsample_type == 'conv':
            downsample_layer = nn.Conv2d
        elif downsample_type == 'resconv':
            downsample_layer = ConvResBlock
        else:
            raise NotImplementedError(
                'downsample layer [{:s}] is not found'.format(downsample_type))
        # upsample
        if upsample_type == 'transconv':
            upsample_layer = nn.ConvTranspose2d
        elif upsample_type == 'interpolate':
            upsample_layer = InterpolateUpsample
        else:
            raise NotImplementedError(
                'upsample layer [{:s}] is not found'.format(upsample_type))

        for i in range(n_downsampling):
            mult = 2**i
            model += [downsample_layer(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        # resnet blocks
        mult = 2**n_downsampling
        bottle_neck = []
        for i in range(n_blocks):
            bottle_neck += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                        activation=activation, norm_layer=norm_layer)]
        if n_attn_g > 0:
            middle = n_blocks//2
            input_size = tuple(map(lambda x: x//mult, input_size))
            from bottleneck_transformer_pytorch import BottleStack
            attn_block = BottleStack(dim=ngf * mult, fmap_size=input_size, dim_out=ngf * mult, num_layers=n_attn_g, proj_factor=proj_factor_g,
                                     downsample=False, heads=heads_g, dim_head=dim_head_g, activation=activation, rel_pos_emb=False)
            bottle_neck.insert(middle, attn_block)
        model += bottle_neck

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [upsample_layer(in_channels=ngf * mult, out_channels=int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf,
                                                   output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)
        self.freeze = False

    def forward(self, input):
        return self.model(input)

    def set_freeze(self, freeze=True):
        if self.freeze == freeze:
            return
        else:
            self.freeze = freeze
        print("The following layers will be freezed:")
        '''Freeze downsample layers'''
        for name, layer in self.model.named_children():
            module_name = layer.__class__.__name__
            if 'ResnetBlock' in module_name or 'BottleStack' in module_name:
                break
            print(name, module_name)
            for param in layer.parameters():
                param.requires_grad = not freeze


class InterpolateUpsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.

    """

    def __init__(self, *args, **kwargs):
        super(InterpolateUpsample, self).__init__()
        self.in_channels = kwargs['in_channels']
        self.out_channels = kwargs['out_channels']
        self.conv1 = nn.Conv2d(
            self.in_channels, self.out_channels, 5, padding=1)
        self.conv2 = nn.Conv2d(
            self.out_channels, self.out_channels, 3, padding=2)
        self.conv_res = nn.Conv2d(
            self.in_channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.in_channels
        x = interpolate(x, scale_factor=2.0, mode="nearest")
        res_x = self.conv_res(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x+res_x


class ConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(
            in_channels, out_channels, 5, padding=2)
        self.conv_res = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        res_x = self.conv_res(x)
        x = self.conv2(x)
        return x+res_x
# Define a resnet block


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        self.output_nc = output_nc

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf), nn.ReLU(True)]
        # downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        # upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf,
                                                   output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i)).nonzero()  # n x 4
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:, 0] + b,
                                         indices[:, 1] + j, indices[:, 2], indices[:, 3]]
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[indices[:, 0] + b, indices[:, 1] +
                                 j, indices[:, 2], indices[:, 3]] = mean_feat
        return outputs_mean


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(
                input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer' +
                            str(j), getattr(netD, 'model'+str(j)))
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j))
                         for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, input_nc=2, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, base_nfft=2048, window=None, min_value=1e-7, mdct_type='4', normalizer=None, getIntermFeat=False, abs_spectro=False):
        super(MultiResolutionDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.base_nfft = base_nfft
        self.window = window
        self.min_value = min_value
        self.mdct = []
        self.normalizer = normalizer
        self.getIntermFeat = getIntermFeat
        self.abs_spectro = abs_spectro

        if mdct_type == '4':
            from .mdct import MDCT4
        elif mdct_type == '2':
            from .mdct import MDCT2
            from dct.dct_native import DCT_2N_native
        else:
            raise NotImplementedError(
                'MDCT type [%s] is not implemented' % mdct_type)

        for i in range(num_D):
            netD = NLayerDiscriminator(
                input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer' +
                            str(j), getattr(netD, 'model'+str(j)))
            else:
                setattr(self, 'layer'+str(i), netD.model)

            if i == 0:
                N = int(self.base_nfft*2)
            else:
                N = int(self.base_nfft//(2**i))
            if mdct_type == '4':
                self.mdct.append(MDCT4(n_fft=N, hop_length=N//2,
                                 win_length=N, window=self.window, center=True))
            elif mdct_type == '2':
                _dct = DCT_2N_native()
                self.mdct.append(MDCT2(n_fft=N, hop_length=N//2, win_length=N,
                                 window=self.window, dct_op=_dct, center=True))

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, waveform):
        result = []
        # FRAME_LENGTH = (BINS-1)*HOP_LENGTH
        bins = waveform.size(-1)//self.base_nfft//2 + 1
        for i in range(self.num_D):
            if i == 0:
                frame_len = int((bins//2-1)*self.base_nfft)
            else:
                N = int(self.base_nfft//(2**i))
                frame_len = int((bins*(2**i)-1)*N)
            len_diff = frame_len - waveform.size(-1)
            if len_diff < 0:
                waveform_ = waveform[..., :len_diff]
            else:
                waveform_ = pad(waveform, (0, len_diff))
            spectro = self.mdct[i](waveform_)
            if self.abs_spectro:
                # [LR, HR/SR, abs(HR/SR)]
                spectro = torch.cat(
                    (spectro, spectro[:, 1, :, :].abs().unsqueeze(1)), dim=1)
            if callable(self.normalizer):
                # [0] avoids multiple return values
                spectro = self.normalizer(spectro)[0]
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(self.num_D-1-i)+'_layer'+str(j))
                         for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(self.num_D-1-i))
            result.append(self.singleD_forward(model, spectro.float()))
        return result

# Defines the PatchGAN discriminator with the specified arguments.


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
