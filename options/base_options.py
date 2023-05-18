import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='label2city', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='pix2pixHD', help='which model to use')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
        self.parser.add_argument('--fp16', action='store_true', default=False, help='train with AMP')
        self.parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
        self.parser.add_argument('--seed', type=int, default=42, help='random seed for reproducing results')
        self.parser.add_argument('--fit_residual', action='store_true', default=False, help='if specified, fit HR-LR than directly fit HR')

        # input/output sizes
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=1024, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--label_nc', type=int, default=0, help='# of input label channels')
        self.parser.add_argument('--input_nc', type=int, default=2, help='# of input spectro channels')
        self.parser.add_argument('--output_nc', type=int, default=1, help='# of output spectro channels')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='./datasets/vctk/train.csv')
        self.parser.add_argument('--evalroot', type=str, default='./datasets/vctk/test.csv')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--explicit_encoding', action='store_true', help='if selected, using trick to encode phase')
        self.parser.add_argument('--alpha', type=float, default=0.6, help='phase encoding factor')
        self.parser.add_argument('--norm_range', type=float, default=(0,1), nargs=2, help='specify the target ditribution range')
        self.parser.add_argument('--abs_norm', action='store_true', help='if selected, assuming the spectrograms are all distributed in a fixed range. Thus instead of normalizing by min and max each by each, normalize by an absolute range.')
        self.parser.add_argument('--src_range', type=float, default=(-5,5), nargs=2, help='specify the source ditribution range. This value is used when --abs_norm is specified.')
        self.parser.add_argument('--arcsinh_transform', action='store_true', help='if selected, using log(G*x+sqrt(((G*x)^2+1))) to compressing the range of input. Do not use this option with --explicit_encoding')
        self.parser.add_argument('--raw_mdct', action='store_true', help='if selected, DO NO transform. Do not use this option with --explicit_encoding|arcsinh_transform')
        self.parser.add_argument('--arcsinh_gain', type=float, default=500, help='gain of arcsinh_trasform input')
        self.parser.add_argument('--add_noise', action='store_true', help='if selected, add some noise to input waveform')
        self.parser.add_argument('--snr', type=float, default=55, help='add noise by SnR (working if --add_noise is selected)')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--upsample_type', type=str, default='transconv', help='selects upsampling layers for netG [transconv|interpolate]')
        self.parser.add_argument('--downsample_type', type=str, default='conv', help='selects upsampling layers for netG [resconv|conv]')
        self.parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG')
        self.parser.add_argument('--n_blocks_global', type=int, default=9, help='number of residual blocks in the global generator network')
        self.parser.add_argument('--n_blocks_attn_g', type=int, default=1, help='number of attention blocks in the global generator network')
        self.parser.add_argument('--proj_factor_g', type=int, default=4, help='projection factor of attention blocks in the global generator network')
        self.parser.add_argument('--dim_head_g', type=int, default=128, help='dim of attention heads in the global generator network')
        self.parser.add_argument('--heads_g', type=int, default=4, help='number of attention heads in the global generator network')
        self.parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--n_blocks_attn_l', type=int, default=0, help='number of attention blocks in the local enhancer network')
        self.parser.add_argument('--proj_factor_l', type=int, default=4, help='projection factor of attention blocks in the local enhancers network')
        self.parser.add_argument('--dim_head_l', type=int, default=128, help='dim of attention heads in the local enhancers network')
        self.parser.add_argument('--heads_l', type=int, default=4, help='number of attention heads in the local enhancers network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')
        self.parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')

        # for instance-wise features
        # self.parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input', default=True)
        # self.parser.add_argument('--instance_feat', action='store_true', help='if specified, add encoded instance features as input')
        # self.parser.add_argument('--label_feat', action='store_true', help='if specified, add encoded label features as input')
        # self.parser.add_argument('--feat_num', type=int, default=3, help='vector length for encoded features')
        # self.parser.add_argument('--load_features', action='store_true', help='if specified, load precomputed feature maps')
        # self.parser.add_argument('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder')
        # self.parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        # self.parser.add_argument('--n_clusters', type=int, default=10, help='number of clusters for features')

        # input mask options
        self.parser.add_argument('--mask', action='store_true', help='mask high freq conponent of lr spectro')
        self.parser.add_argument('--smooth', type=float, default=0.0, help='smooth the edge of the sr and lr')
        self.parser.add_argument('--mask_hr', action='store_true', help='mask high freq conponent of hr spectro')
        self.parser.add_argument('--mask_mode', type=str, default=None, help='[None|mode0|mode1]')
        self.parser.add_argument('--min_value', type=float, default=1e-7, help='minimum value to cutoff the spectrogram')

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
