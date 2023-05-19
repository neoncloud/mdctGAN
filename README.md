# mdctGAN: Taming transformer-based GAN for speech super-resolution with Modified DCT spectra
[arXiv](https://arxiv.org/abs/2305.11104)
## Requirements
* bottleneck_transformer_pytorch==0.1.4
* dominate
* einops
* matplotlib
* numpy
* Pillow
* scipy
* torch
* torchaudio
* torchvision

## Data Preparation
Firstly, Prepare your dataset index file like this (VCTK dataset example):
```
wav48/p250/p250_328.wav
wav48/p310/p310_345.wav
wav48/p227/p227_020.wav
wav48/p285/p285_050.wav
wav48/p248/p248_011.wav
wav48/p246/p246_030.wav
wav48/p247/p247_191.wav
wav48/p287/p287_127.wav
wav48/p334/p334_220.wav
wav48/p340/p340_414.wav
wav48/p236/p236_231.wav
wav48/p301/p301_334.wav
...
```
Save it to the root directory of your dataset and the program will splice the parent folder of index file with the relative path of the records in the file. You can also find the index file used in our experiments in `data/train.csv`.

Secondly, for excessively long audio file, we recommend that you split it into smaller segments and remove long gaps. Other than this no other pre-processing is required, the program will automatically sample a random section from the longer audio file.
## Train
Modify & run `sh train.sh`. Detailed explanation of args can be found in `options/base_options.py` and `options/train_options.py`


| Parameter Name       | Description                                                                                      |
|----------------------|--------------------------------------------------------------------------------------------------|
| --name               | Name of the experiment. It decides where to store samples and models.                            |
| **--dataroot**       | Path to your train set csv file.                                                                 |
| **--evalroot**       | Path to your eval set csv file.                                                                  |
| **--lr_sampling_rate**   | Input Low-res sampling rate. It will be automatically resampled to this value.               |
| **--sr_sampling_rate**   | Target super-resolution sampling rate.                                                       |
| --fp16               | Train with Automatic Mixed Precision (AMP).                                                      |
| --nThreads           | Number of threads for loading data.                                                              |
| --lr                 | Initial learning rate for the Adam optimizer.                                                    |
| --arcsinh_transform  | Use $\log(x+\sqrt{x^2+1})$ to compress the range of input.                                       |
| --abs_spectro        | Use the absolute value of the spectrogram.                                                       |
| --arcsinh_gain       | Gain parameter for the arcsinh_transform.                                                        |
| --center             | Centered MDCT.                                                                                   |
| --norm_range         | Specify the target distribution range.                                                           |
| --abs_norm           | Assume the spectrograms are all distributed in a fixed range. Normalize by an absolute range.    |
| --src_range          | Specify the source distribution range. Used when --abs_norm is specified.                        |
| --netG               | Select the model to use for netG.                                                                |
| --ngf                | Number of generator filters in the first conv layer.                                             |
| --n_downsample_global| Number of downsampling layers in netG.                                                           |
| --n_blocks_global    | Number of residual blocks in the global generator network.                                       |
| --n_blocks_attn_g    | Number of attention blocks in the global generator network.                                      |
| --dim_head_g         | Dimension of attention heads in the global generator network.                                    |
| --heads_g            | Number of attention heads in the global generator network.                                       |
| --proj_factor_g      | Projection factor of attention blocks in the global generator network.                           |
| --n_blocks_local     | Number of residual blocks in the local enhancer network.                                         |
| --n_blocks_attn_l    | Number of attention blocks in the local enhancer network.                                        |
| --fit_residual       | If specified, fit $HR-LR$ than directly fit $HR$.                                                |
| --upsample_type      | Select upsampling layers for netG. Supported options: interpolate, transconv.                    |
| --downsample_type    | Select downsampling layers for netG. Supported options: resconv, conv.                           |
| --num_D              | Number of discriminators to use.                                                                 |
| --eval_freq          | Frequency of evaluating metrics.                                                                 |
| --save_latest_freq   | Frequency of saving the latest results.                                                          |
| --save_epoch_freq    | Frequency of saving checkpoints at the end of epochs.                                            |
| --display_freq       | Frequency of showing training results on screen.                                                 |
| --tf_log             | If specified, use TensorBoard logging. Requires TensorFlow installed.                            |

## Evaluate & Generate audio
Modify & run `sh gen_audio.sh`.

## Acknowledgement
This code repository refers heavily to the [official pix2pixHD implementation](https://github.com/NVIDIA/pix2pixHD). Also, this work is based on an improved version of my undergraduate Final Year Project, see: [pix2pixHDAudioSR](https://github.com/neoncloud/pix2pixHDAudioSR)
