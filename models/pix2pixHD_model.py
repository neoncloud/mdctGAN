from typing import Dict
import numpy as np
import torch
from util.image_pool import ImagePool
from util.spectro_img import compute_visuals
from util.util import kbdwin
from .base_model import BaseModel
from . import networks
from .mdct import MDCT4, IMDCT4
import torchaudio.functional as aF
# from torch_audiomentations import Shift


class Audio2MDCT(torch.nn.Module):
    def __init__(self, opt) -> None:
        super(Audio2MDCT, self).__init__()
        opt_dict = vars(opt)
        for k, v in opt_dict.items():
            setattr(self, k, v)
        # define mdct and imdct
        self.device = 'cuda' if len(self.gpu_ids) > 0 else 'cpu'
        self.up_ratio = self.hr_sampling_rate / self.lr_sampling_rate
        self.window = kbdwin(self.win_length).to(self.device)
        self.min_value = opt.min_value
        #self._dct = DCT_2N_native()
        self._mdct = MDCT4(n_fft=self.n_fft, hop_length=self.hop_length,
                           win_length=self.win_length, window=self.window, device=self.device)
        #self._idct = IDCT_2N_native()
        self._imdct = IMDCT4(n_fft=self.n_fft, hop_length=self.hop_length,
                             win_length=self.win_length, window=self.window, device=self.device)

    def to_spectro(self, audio: torch.Tensor, mask: bool = False, mask_size: int = -1):
        # Forward Transformation (MDCT)
        spectro, frames = self._mdct(audio, True)
        spectro = spectro.unsqueeze(1)
        pha = torch.sign(spectro)

        log_spectro, audio_max, audio_min, mean, std = self.normalize(spectro)

        #log_audio = (log_audio-mean)/std
        # Deprecated, for there already has been Instance Norm.

        # if explicit_encoding:
        #     # multiply phase with log magnitude
        #     log_audio = (log_audio-audio_min)/(audio_max-audio_min)
        #     # log_audio @ [0,1]
        #     log_audio = log_audio*pha
        #     # log_audio @ [-1,1], double peak
        if not self.explicit_encoding:  # TODO
            _noise = torch.randn(pha.size(), device=self.device)
            _noise_min = _noise.min()
            _noise_max = _noise.max()
            _noise = (_noise - _noise_min)/(_noise_max - _noise_min)
            pha = pha*_noise
        # log_audio @ [-1,1], singal peak

        if mask:
            # mask the lr spectro so that it does not learn from garbage infomation
            size = log_spectro.size()
            if mask_size == -1:
                mask_size = int(size[3]*(1-1/self.up_ratio))

            # fill the blank mask with noise
            _noise = torch.randn(
                size[0], size[1], size[2], mask_size, device=self.device)
            _noise_min = _noise.min()
            _noise_max = _noise.max()

            if self.fit_residual:
                _noise = torch.zeros(
                    size[0], size[1], size[2], mask_size, device=self.device)
            else:
                # fill empty with randn noise, single peak, centered at 0
                _noise = _noise/(_noise_max - _noise_min)

            log_spectro = torch.cat(
                (
                    log_spectro[:, :, :, :-mask_size],
                    _noise
                ), dim=3)
        return log_spectro.float(), pha, {'max': audio_max, 'min': audio_min, 'mean': mean, 'std': std, 'frames': frames}

    def normalize(self, spectro):
        if self.explicit_encoding:
            neg = 0.5*(torch.abs(spectro)-spectro)
            pos = spectro+neg
            log_spectro = torch.cat(
                (
                    aF.amplitude_to_DB(
                        self.alpha*pos+(1-self.alpha)*neg, 20.0, self.min_value, 1.0),
                    aF.amplitude_to_DB(
                        (1-self.alpha)*pos+self.alpha*neg, 20.0, self.min_value, 1.0)
                ),
                dim=1,
            )
        elif self.arcsinh_transform:
            Gain = self.arcsinh_gain
            spectro = Gain*spectro
            log_spectro = torch.arcsinh(
                spectro)/torch.log(torch.tensor(10.0))
            #log_spectro = torch.cat((log_spectro, log_spectro.abs()), dim=1)
        elif self.raw_mdct:
            log_spectro = spectro
        else:
            log_spectro = aF.amplitude_to_DB(
                (torch.abs(spectro) + self.min_value), 20.0, self.min_value, 1.0)

        mean = log_spectro.mean().float()
        std = log_spectro.var().sqrt().float()
        if not self.abs_norm:
            audio_max = log_spectro.flatten(-2).max(dim=-
                                                    1).values[:, :, None, None].float()
            audio_min = log_spectro.flatten(-2).min(dim=-
                                                    1).values[:, :, None, None].float()
        else:
            audio_min = torch.tensor([self.src_range[0]], device=self.device)[
                None, None, None, :]
            audio_max = torch.tensor([self.src_range[1]], device=self.device)[
                None, None, None, :]
        log_spectro = (log_spectro-audio_min)/(audio_max-audio_min)
        log_spectro = log_spectro * \
            (self.norm_range[1]-self.norm_range[0]
             )+self.norm_range[0]

        return log_spectro, audio_max, audio_min, mean, std

    def denormalize(self, log_spectro: torch.Tensor, min: torch.Tensor, max: torch.Tensor):
        log_spectro = (
            log_spectro.to(torch.float64)-self.norm_range[0])/(self.norm_range[1]-self.norm_range[0])
        log_spectro = log_spectro*(max-min)+min
        #log_mag = log_mag*norm_param['std']+norm_param['mean']
        if self.arcsinh_transform:
            return torch.sinh(log_spectro*torch.log(torch.tensor(10.0)))/self.arcsinh_gain
        if self.raw_mdct:
            return log_spectro
        else:
            return aF.DB_to_amplitude(log_spectro, 10.0, 0.5)-self.min_value

    def to_audio(self, log_spectro: torch.Tensor, norm_param: Dict[str, torch.Tensor], pha: torch.Tensor):
        spectro = self.denormalize(
            log_spectro, norm_param['min'], norm_param['max'])
        if self.explicit_encoding:
            spectro = (spectro[..., 0, :, :] -
                       spectro[..., 1, :, :])/(2*self.alpha-1)
        elif self.arcsinh_transform:
            pass
        elif self.raw_mdct:
            pass
        else:
            if self.up_ratio > 1:
                size = pha.size(-2)
                psudo_pha = 2 * \
                    torch.randint(low=0, high=2, size=pha.size(),
                                  device=self.device)-1
                pha = torch.cat((pha[..., :int(size*(1/self.up_ratio)), :],
                                psudo_pha[..., int(size*(1/self.up_ratio)):, :]), dim=-2)
                spectro = spectro*pha

        if self.explicit_encoding:
            audio, _ = self._imdct(spectro)
        else:
            audio, _ = self._imdct(spectro.squeeze(1))
        return audio

    def to_frames(self, log_spectro, norm_param):
        spectro = self.denormalize(
            log_spectro, norm_param['min'], norm_param['max'])
        if self.explicit_encoding:
            spectro = (spectro[..., 0, :, :] -
                       spectro[..., 1, :, :])/(2*self.alpha-1)
        _, frames = self._imdct(spectro.squeeze(), True)
        return frames

    # def audio_rand_shift(self, lr, hr, sr, n: int = 20):
    #     audio = torch.stack((lr, hr, sr), dim=1)
    #     shift = Shift(min_shift=-n//2, max_shift=n//2,
    #                   shift_unit='samples', rollover=False)
    #     return shift(audio).chunk(3, dim=1)

    def norm_frames(self, frames):
        frames = aF.amplitude_to_DB(
            torch.abs(frames), 20, self.min_value, 1)
        frames = frames - frames.min()
        frames = frames / frames.max()
        return frames * (self.norm_range[1]-self.norm_range[0]) + self.norm_range[0]

    def forward(self, lr_audio: torch.Tensor):
        # low-res audio for training
        with torch.no_grad():
            lr_spectro, lr_pha, lr_norm_param = self.to_spectro(
                lr_audio, mask=self.mask)
        return lr_spectro, lr_pha, lr_norm_param

    def hr_forward(self, hr_audio: torch.Tensor):
        # high-res audio for training
        with torch.no_grad():
            hr_spectro, hr_pha, hr_norm_param = self.to_spectro(hr_audio, mask=self.mask_hr, mask_size=int(
                self.n_fft*(1-self.sr_sampling_rate/self.hr_sampling_rate)//2))

        return hr_spectro, hr_pha, hr_norm_param


class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, use_match_loss, use_time_loss, use_mr_loss, use_shift_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, use_match_loss, use_time_loss,
                 use_time_loss, use_time_loss, use_mr_loss, use_mr_loss, use_mr_loss, use_shift_loss, True, True)

        def loss_filter(g_gan, g_gan_feat, g_vgg, g_mat, g_gan_t, d_real_t, d_fake_t, g_gan_mr, d_real_mr, d_fake_mr, g_shift, d_real, d_fake):
            return [l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, g_mat, g_gan_t, d_real_t, d_fake_t, g_gan_mr, d_real_mr, d_fake_mr, g_shift, d_real, d_fake), flags) if f]
        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        opt_dict = vars(opt)
        for k, v in opt_dict.items():
            setattr(self, k, v)
        # if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
        #     torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        # self.use_features = opt.instance_feat or opt.label_feat
        # self.gen_features = self.use_features and not self.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        self.preprocess = Audio2MDCT(opt)

        # define networks
        # Generator network
        # set freeze network
        self.freeze = opt.freeze_g_d or opt.freeze_g_u or opt.freeze_l_d or opt.freeze_l_u
        netG_input_nc = input_nc
        # if not opt.no_instance:
        #     netG_input_nc += 1
        # if self.use_features:
        #     netG_input_nc += opt.feat_num
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids, upsample_type=opt.upsample_type, downsample_type=opt.downsample_type, input_size=(opt.bins, opt.n_fft//2), n_attn_g=opt.n_blocks_attn_g, n_attn_l=opt.n_blocks_attn_l, proj_factor_g=opt.proj_factor_g, heads_g=opt.heads_g, dim_head_g=opt.dim_head_g, proj_factor_l=opt.proj_factor_l, heads_l=opt.heads_l, dim_head_l=opt.dim_head_l)
        self.netG.set_freeze(opt.freeze_g_d, opt.freeze_g_u,
                             opt.freeze_l_d, opt.freeze_l_u)
        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            # if not opt.no_instance:
            #     netD_input_nc += 1
            if self.abs_spectro and self.arcsinh_transform:
                # discriminate on [LR, HR/SR, abs(HR/SR)]
                #netD_input_nc += 1
                pass
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            # if opt.use_hifigan_D:
            #     from .ParallelWaveGAN.parallel_wavegan.models.hifigan import HiFiGANMultiScaleMultiPeriodDiscriminator
            #     self.hifigan_D = HiFiGANMultiScaleMultiPeriodDiscriminator().to(self.device)
            # if opt.use_time_D:
            #     self.time_D = networks.define_D(
            #         2, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, False, gpu_ids=self.gpu_ids)
            # if opt.use_multires_D:
            #     self.multires_D = networks.define_MR_D(opt.ndf, opt.n_layers_D, netD_input_nc, opt.norm, use_sigmoid,
            #                                            opt.num_mr_D, gpu_ids=self.gpu_ids, base_nfft=opt.n_fft, window=kbdwin, min_value=opt.min_value, mdct_type='4', normalizer=self.preprocess.normalize, getIntermFeat=not opt.no_ganFeat_loss, abs_spectro=opt.abs_spectro)

        # Encoder network
        # if self.gen_features:
        #     self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder',
        #                                   opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)
        if self.verbose:
            print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(
                    self.netD, 'D', opt.which_epoch, pretrained_path)
            # if self.gen_features:
            #     self.load_network(
            #         self.netE, 'E', opt.which_epoch, pretrained_path)
            # if opt.use_hifigan_D:
            #     self.load_network(self.hifigan_D, 'hifigan_D',
            #                       opt.which_epoch, pretrained_path)
            # if opt.use_time_D:
            #     self.load_network(self.time_D, 'time_D',
            #                       opt.which_epoch, pretrained_path)
            # if opt.use_multires_D:
            #     self.load_network(self.multires_D, 'multires_D',
            #                       opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError(
                    "Fake Pool Not Implemented for MultiGPU")
            # pools that store previously generated samples
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            self.limit_aux_loss = False
            # if self.use_time_D:
            #     self.time_D_count = opt.time_D_count
            #     self.time_D_count_max = opt.time_D_count_max

            # if self.use_match_loss:
            #     self.match_loss_count = opt.match_loss_count
            #     self.match_loss_count_max = opt.match_loss_count_max
            #     self.match_loss_thres = opt.match_loss_thres

            # define loss functions
            self.loss_filter = self.init_loss_filter(
                not opt.no_ganFeat_loss, False, False, False, False, False)

            self.criterionGAN = networks.GANLoss(
                use_lsgan=not opt.no_lsgan, device=self.device)
            self.criterionFeat = torch.nn.L1Loss()
            # if opt.use_match_loss:
            #     self.criterionMatch = torch.nn.MSELoss()
            # if opt.use_shifted_match:
            #     self.criterionSMatch = torch.nn.L1Loss()
            # if not opt.no_vgg_loss:
            #     self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'G_mat', 'G_GAN_t',
                                               'D_real_t', 'D_fake_t', 'G_GAN_mr', 'D_real_mr', 'D_fake_mr', 'G_shift', 'D_real', 'D_fake')

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                finetune_list = set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [value]
                        finetune_list.add(key.split('.')[0])
                print('------------- Only training the local enhancer network (for %d epochs) ------------' %
                      opt.niter_fix_global)
                print('The layers that are finetuned are ',
                      sorted(finetune_list))
            else:
                params = list(self.netG.parameters())
            # if self.gen_features:
            #     params += list(self.netE.parameters())
            print('Total number of parameters of G: %d' %
                  (sum([param.numel() for param in params])))
            self.optimizer_G = torch.optim.Adam(
                filter(lambda p: p.requires_grad, params), lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            # if self.use_hifigan_D:
            #     params += list(self.hifigan_D.parameters())
            # if self.use_time_D:
            #     params += list(self.time_D.parameters())
            # if self.use_multires_D:
            #     params += list(self.multires_D.parameters())
            print('Total number of parameters of D: %d' %
                  (sum([param.numel() for param in params])))
            self.optimizer_D = torch.optim.Adam(
                params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def discriminate_F(self, input_label, test_image, use_pool=False):
        '''Frequency domain discriminator'''
        # notice the test_image is detached, hence it wont backward to G networks.
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    # def discriminate_time_D(self, label_spectro, test_spectro):
    #     '''Time domain discriminator'''
    #     # notice the test_image is detached, hence it wont backward to G networks.
    #     input_concat = torch.cat((label_spectro, test_spectro.detach()), dim=1)
    #     return self.time_D.forward(input_concat)

    # def discriminate_hifi(self, input, norm_param=None, pha=None, is_spectro=True):
    #     '''Time domain discriminator using hifi_gan_D'''
    #     # input shape [B 1 T]
    #     if is_spectro:
    #         waveform = self.preprocess.to_audio(
    #             input, norm_param=norm_param, pha=pha).squeeze().unsqueeze(1)
    #     else:
    #         waveform = input.unsqueeze(1)
    #     if self.fp16:
    #         waveform = waveform.half()
    #     return self.hifigan_D.forward(waveform)

    def forward(self, lr_audio, hr_audio):
        # Encode Inputs
        lr_spectro, lr_pha, lr_norm_param = self.preprocess.forward(lr_audio)
        hr_spectro, hr_pha, hr_norm_param = self.preprocess.hr_forward(
            hr_audio)
        #### G Forward ####
        if self.abs_spectro and self.arcsinh_transform:
            lr_input = lr_spectro.abs()*2+self.norm_range[0]
            lr_input = torch.cat((lr_spectro, lr_input), dim=1)
        else:
            lr_input = lr_spectro
        sr_spectro = self.netG.forward(lr_input)
        #### G Forward ####
        if self.fit_residual:
            sr_spectro = sr_spectro+lr_spectro
        if self.explicit_encoding:
            sr_pha = torch.sign(
                sr_spectro[:, 0, :, :]-sr_spectro[:, 1, :, :]).unsqueeze(1)
        else:
            sr_pha = None
        return sr_spectro, sr_pha, hr_spectro, hr_pha, hr_norm_param, lr_spectro, lr_pha, lr_norm_param

    def _forward(self, lr_audio, hr_audio, infer=False):
        sr_spectro, sr_pha, hr_spectro, hr_pha, hr_norm_param, lr_spectro, _, lr_norm_param = self.forward(
            lr_audio, hr_audio)
        # Fake Detection and Loss
        if self.abs_spectro and self.arcsinh_transform:
            sr_input = torch.cat(
                (sr_spectro, sr_spectro.abs()*2+self.norm_range[0]), dim=1)
            hr_input = torch.cat(
                (hr_spectro, hr_spectro.abs()*2+self.norm_range[0]), dim=1)
        else:
            sr_input = sr_spectro
            hr_input = hr_spectro

        pred_fake_pool = self.discriminate_F(
            lr_spectro, sr_input, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = self.discriminate_F(lr_spectro, hr_input)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        # make a new input pair without detaching, the loss will hence backward to G
        pred_fake = self.netD.forward(
            torch.cat((lr_spectro, sr_input), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.no_ganFeat_loss:
            feat_weights = 4.0 / (self.n_layers_D + 1)
            D_weights = 1.0 / self.num_D
            for i in range(self.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(
                            pred_fake[i][j], pred_real[i][j].detach()) * self.lambda_feat

        # (deprecated) Time shifting loss
        loss_shift = 0
        # if self.use_shifted_match:
        #     n = 20
        #     sr_audio = self.preprocess.to_audio(
        #         sr_spectro, norm_param=lr_norm_param, pha=torch.empty(1)).squeeze(1).to(lr_audio.dtype).squeeze()
        #     _, _hr_audio, _sr_audio = self.preprocess.audio_rand_shift(
        #         lr_audio.to(self.device).squeeze(), hr_audio.to(self.device).squeeze(), sr_audio, n)
        #     _hr_audio, _sr_audio = _hr_audio.squeeze(), _sr_audio.squeeze()

        #     _sr_spectro, _, _ = self.preprocess.to_spectro(_sr_audio)
        #     _hr_spectro, _, _ = self.preprocess.to_spectro(_hr_audio)

        #     loss_shift = self.criterionSMatch(
        #         _sr_spectro, _hr_spectro.detach())

        # (deprecated) Multi_Res Discriminator loss
        loss_G_GAN_mr = 0
        loss_D_real_mr = 0
        loss_D_fake_mr = 0
        # if self.use_multires_D:
        #     lr_audio = lr_audio.to(self.device).unsqueeze(1)
        #     hr_audio = hr_audio.to(self.device).unsqueeze(1)
        #     sr_audio = self.preprocess.to_audio(
        #         sr_spectro, norm_param=lr_norm_param, pha=torch.empty(1)).squeeze(1).to(lr_audio.dtype)
        #     if self.explicit_encoding:
        #         sr_audio = np.sqrt(self.up_ratio-1)*sr_audio

        #     # Fake Detection and Loss
        #     pred_fake_pool_mr = self.multires_D.forward(
        #         torch.cat((lr_audio, sr_audio.detach()), dim=1))
        #     loss_D_fake_mr = self.criterionGAN(
        #         pred_fake_pool_mr, False)*self.lambda_mr

        #     # Real Detection and Loss
        #     pred_real_mr = self.multires_D.forward(
        #         torch.cat((lr_audio, hr_audio.detach()), dim=1))
        #     loss_D_real_mr = self.criterionGAN(
        #         pred_real_mr, True)*self.lambda_mr

        #     # GAN loss (Fake Passability Loss)
        #     # make a new input pair without detaching, the loss will hence backward to G
        #     pred_fake_mr = self.multires_D.forward(
        #         torch.cat((lr_audio, sr_audio), dim=1))
        #     loss_G_GAN_mr = self.criterionGAN(
        #         pred_fake_mr, True)*self.lambda_mr

        # (deprecated) Time domain GAN, including hifi_gan_D and D on frames
        loss_G_GAN_time = 0
        loss_D_real_time = 0
        loss_D_fake_time = 0
        # if self.use_hifigan_D:
        #     scaler = self.time_D_count/self.time_D_count_max
        #     pred_fake_time = self.discriminate_hifi(
        #         sr_spectro, norm_param=lr_norm_param, is_spectro=True)
        #     loss_G_GAN_time += self.criterionGAN(
        #         pred_fake_time, True)*self.lambda_time*scaler
        #     pred_real_time = self.discriminate_hifi(
        #         hr_audio.detach(), is_spectro=False)
        #     loss_D_real_time += self.criterionGAN(
        #         pred_real_time, True)*self.lambda_time*scaler
        #     _pred_fake_time = self.discriminate_hifi(
        #         sr_spectro.detach(), norm_param=lr_norm_param, is_spectro=True)
        #     loss_D_fake_time += self.criterionGAN(
        #         _pred_fake_time, False)*self.lambda_time*scaler
        # if self.use_time_D:
        #     scaler = self.time_D_count/self.time_D_count_max
        #     lr_frames = lr_norm_param['frames'][:, None, :, :]
        #     hr_frames = hr_norm_param['frames'][:, None, :, :]
        #     sr_frames = self.preprocess.to_frames(sr_spectro, lr_norm_param)[:, None, :, :].to(
        #         torch.half if self.fp16 else torch.float)

        #     lr_frames = self.preprocess.norm_frames(lr_frames)
        #     hr_frames = self.preprocess.norm_frames(hr_frames)
        #     sr_frames = self.preprocess.norm_frames(sr_frames)

        #     pred_fake_frames = self.discriminate_time_D(lr_frames, sr_frames)
        #     loss_D_fake_time += self.criterionGAN(
        #         pred_fake_frames, False)*self.lambda_time*scaler

        #     pred_real_frames = self.discriminate_time_D(lr_frames, hr_frames)
        #     loss_D_real_time += self.criterionGAN(
        #         pred_real_frames, True)*self.lambda_time*scaler

        #     _pred_fake_frames = self.time_D.forward(
        #         torch.cat((lr_frames, sr_frames), dim=1))
        #     loss_G_GAN_time += self.criterionGAN(
        #         _pred_fake_frames, True)*self.lambda_time*scaler

        # (deprecated) VGG feature matching loss
        loss_G_VGG = 0

        # (deprecated) Frame matching loss
        # The overlapped part of frames should be the same
        loss_G_match = 0
        # if self.use_match_loss:
        #     scaler = self.match_loss_count/self.match_loss_count_max
        #     half = self.win_length//2
        #     sr_frames = self.preprocess.to_frames(sr_spectro, lr_norm_param)
        #     # if self.fit_residual:
        #     #     lr_frames = lr_norm_param['frames'].unsqueeze(1)
        #     #     sr_frames = sr_frames-lr_frames
        #     a = sr_frames[..., :-1, half:]*self.window[:half]
        #     # a = aF.amplitude_to_DB(
        #     # torch.abs(a), 20, self.min_value, 1)
        #     b = sr_frames[..., 1:, :half]*self.window[half:]
        #     # b = aF.amplitude_to_DB(
        #     # torch.abs(b), 20, self.min_value, 1)
        #     loss_G_match = torch.mean(self.criterionMatch(
        #         a, b.detach())) * self.lambda_mat * scaler
        #     if self.limit_aux_loss:
        #         if loss_G_match > self.match_loss_thres:
        #             print('Match loss too large:', loss_G_match)
        #             loss_G_match = 0

        # Register current samples
        if self.arcsinh_transform:
            self.current_lable = (lr_spectro.detach().cpu(
            )-self.norm_range[0])/(self.norm_range[1]-self.norm_range[0])
            self.current_lable = (self.current_lable*(lr_norm_param['max'].cpu(
            )-lr_norm_param['min'].cpu())+lr_norm_param['min'].cpu()).numpy()[0, 0, :, :]

            min_val = hr_norm_param['min'].cpu().numpy()[0, 0, :, :]
            max_val = hr_norm_param['max'].cpu().numpy()[0, 0, :, :]
            self.current_generated = (sr_spectro.detach().cpu(
            )-self.norm_range[0])/(self.norm_range[1]-self.norm_range[0])
            self.current_generated = np.clip((self.current_generated*(lr_norm_param['max'].cpu(
            )-lr_norm_param['min'].cpu())+lr_norm_param['min'].cpu()).numpy()[0, 0, :, :], min_val, max_val)

            self.current_real = (hr_spectro.detach().cpu(
            )-self.norm_range[0])/(self.norm_range[1]-self.norm_range[0])
            self.current_real = (self.current_real*(hr_norm_param['max'].cpu(
            )-hr_norm_param['min'].cpu())+hr_norm_param['min'].cpu()).numpy()[0, 0, :, :]
        else:
            self.current_lable = lr_spectro.detach().cpu().numpy()[0, 0, :, :]
            self.current_generated = sr_spectro.detach().cpu().numpy()[
                0, 0, :, :]
            self.current_real = hr_spectro.detach().cpu().numpy()[0, 0, :, :]

        # Additional visuals
        if self.explicit_encoding:
            self.current_lable = 0.5 * \
                (lr_spectro[0, 0, :, :]+lr_spectro[0, 1, :, :]
                 ).detach().cpu().numpy()
            self.current_generated = 0.5 * \
                (sr_spectro[0, 0, :, :]+sr_spectro[0, 1, :, :]
                 ).detach().cpu().numpy()
            self.current_real = 0.5 * \
                (hr_spectro[0, 0, :, :]+hr_spectro[0, 1, :, :]
                 ).detach().cpu().numpy()
            if self.input_nc >= 2:
                self.current_lable_pha = (
                    hr_pha-sr_pha).detach().cpu().numpy()[0, 0, :, :]
                self.current_generated_pha = sr_pha.detach().cpu().numpy()[
                    0, 0, :, :]
                self.current_real_pha = hr_pha.detach().cpu().numpy()[
                    0, 0, :, :]
        else:
            self.current_lable_pha = None
            self.current_generated_pha = None
            self.current_real_pha = None

        # Only return the fake_B image if necessary to save BW
        return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_match, loss_G_GAN_time, loss_D_real_time, loss_D_fake_time, loss_G_GAN_mr, loss_D_real_mr, loss_D_fake_mr, loss_shift, loss_D_real, loss_D_fake), torch.empty if not infer else sr_spectro]

    def inference(self, lr_audio):
        # Encode Inputs
        with torch.no_grad():
            lr_spectro, lr_pha, lr_norm_param = self.preprocess.forward(
                lr_audio)

            if self.abs_spectro and self.arcsinh_transform:
                lr_input = lr_spectro.abs()*2+self.norm_range[0]
                lr_input = torch.cat((lr_spectro, lr_input), dim=1)
            else:
                lr_input = lr_spectro

            sr_spectro = self.netG.forward(lr_input)
            if self.fit_residual:
                up_ratio = self.preprocess.up_ratio
                lr_part = int(sr_spectro.size(-1)/up_ratio)
                sr_spectro[..., :lr_part] *= 1e-3
                sr_spectro = sr_spectro+lr_spectro
        sr_audio = self.preprocess.to_audio(sr_spectro, lr_norm_param, lr_pha)

        return sr_spectro, sr_audio, lr_pha, lr_norm_param, lr_spectro

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        # if self.gen_features:
        #     self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)
        # if self.use_hifigan_D:
        #     self.save_network(self.hifigan_D, 'hifigan_D',
        #                       which_epoch, self.gpu_ids)
        # if self.use_time_D:
        #     self.save_network(self.time_D, 'time_D', which_epoch, self.gpu_ids)
        # if self.use_multires_D:
        #     self.save_network(self.multires_D, 'multires_D',
        #                       which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        # if self.gen_features:
        #     params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(
            params, lr=self.lr, betas=(self.beta1, 0.999))
        if self.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.lr / self.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    # def update_match_loss_scaler(self):
    #     if self.match_loss_count < self.match_loss_count_max:
    #         self.match_loss_count += 1

    # def update_time_D_loss_scaler(self):
    #     if self.time_D_count < self.time_D_count_max:
    #         self.time_D_count += 1

    def get_current_visuals(self):
        lable_sp, lable_hist, _ = compute_visuals(
            sp=self.current_lable, abs=self.abs_spectro)
        _, _, lable_pha = compute_visuals(pha=self.current_lable_pha)
        generated_sp, generated_hist, _ = compute_visuals(
            sp=self.current_generated, abs=self.abs_spectro)
        _, _, generated_pha = compute_visuals(pha=self.current_generated_pha)
        real_sp, real_hist, _ = compute_visuals(
            sp=self.current_real, abs=self.abs_spectro)
        _, _, real_pha = compute_visuals(pha=self.current_real_pha)
        if self.current_lable_pha is not None:
            return {'lable_spectro':        lable_sp,
                    'generated_spectro':    generated_sp,
                    'real_spectro':         real_sp,
                    'lable_hist':           lable_hist,
                    'generated_hist':       generated_hist,
                    'real_hist':            real_hist,
                    'lable_pha':            lable_pha,
                    'generated_pha':        generated_pha,
                    'real_pha':             real_pha}
        else:
            return {'lable_spectro':        lable_sp,
                    'generated_spectro':    generated_sp,
                    'real_spectro':         real_sp,
                    'lable_hist':           lable_hist,
                    'generated_hist':       generated_hist,
                    'real_hist':            real_hist, }


class InferenceModel(Pix2PixHDModel):
    def forward(self, lr_audio):
        return self.inference(lr_audio)
