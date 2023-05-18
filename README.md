# mdctGAN: Taming transformer-based GAN for speech super-resolution with Modified DCT spectra

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

## Train
Modify & run `sh train.sh`. Detailed explanation of args can be found in `options/base_options.py` and `options/train_options.py`


| Parameter Name       | Description                                                                                      |
|----------------------|--------------------------------------------------------------------------------------------------|
| --name               | Name of the experiment. It decides where to store samples and models.                            |
| --batchSize          | Input batch size.                                                                                |
| --fp16               | Train with Automatic Mixed Precision (AMP).                                                      |
| --nThreads           | Number of threads for loading data.                                                              |
| --lr                 | Initial learning rate for the Adam optimizer.                                                    |
| --arcsinh_transform  | Use log(G*x+sqrt(((G*x)^2+1))) to compress the range of input. Do not use with --explicit_encoding.|
| --abs_spectro        | Use the absolute value of the spectrogram.                                                       |
| --arcsinh_gain       | Gain parameter for the arcsinh_transform.                                                        |
| --center             | Centered MDCT.                                                                                    |
| --norm_range         | Specify the target distribution range.                                                           |
| --smooth             | Smooth the edge of the SR and LR.                                                                 |
| --abs_norm           | Assume the spectrograms are all distributed in a fixed range. Normalize by an absolute range.     |
| --src_range          | Specify the source distribution range. Used when --abs_norm is specified.                         |
| --netG               | Select the model to use for netG.                                                                 |
| --ngf                | Number of generator filters in the first conv layer.                                              |
| --n_downsample_global| Number of downsampling layers in netG.                                                            |
| --n_blocks_global    | Number of residual blocks in the global generator network.                                        |
| --n_blocks_attn_g    | Number of attention blocks in the global generator network.                                       |
| --dim_head_g         | Dimension of attention heads in the global generator network.                                     |
| --heads_g            | Number of attention heads in the global generator network.                                        |
| --proj_factor_g      | Projection factor of attention blocks in the global generator network.                            |
| --n_blocks_local     | Number of residual blocks in the local enhancer network.                                          |
| --n_blocks_attn_l    | Number of attention blocks in the local enhancer network.                                         |
| --fit_residual       | If specified, fit HR-LR than directly fit HR.                                                     |
| --upsample_type      | Select upsampling layers for netG. Supported options: interpolate, transconv.                    |
| --downsample_type    | Select downsampling layers for netG. Supported options: resconv, conv.                            |
| --niter              | Number of iterations at the starting learning rate.                                               |
| --niter_decay        | Number of iterations to linearly decay the learning rate to zero.                                 |
| --num_D              | Number of discriminators to use.                                                                 |
| --eval_freq          | Frequency of evaluating metrics.                                                                 |
| --save_latest_freq   | Frequency of saving the latest results.                                                          |
| --save_epoch_freq    | Frequency of saving checkpoints at the end of epochs.                                             |
| --display_freq       | Frequency of showing training results on screen.                                                  |
| --tf_log             | If specified, use TensorBoard logging. Requires TensorFlow installed.                             |

## Evaluate & Generate audio
Modify & run `sh gen_audio.sh`.

## Acknowledgement
This code repository refers heavily to the [official pix2pixHD implementation](https://github.com/NVIDIA/pix2pixHD). Also, this work is based on an improved version of my undergraduate graduation design, see: [pix2pixHDAudioSR](https://github.com/neoncloud/pix2pixHDAudioSR)