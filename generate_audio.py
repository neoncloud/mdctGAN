from util.util import compute_matrics
import torch
import torchaudio
import os

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.spectro_img import compute_visuals

# Initilize the setup
opt = TrainOptions().parse()
visualizer = Visualizer(opt)
data_loader = CreateDataLoader(opt)
dataset = data_loader.get_train_dataloader()
dataset_size = len(data_loader)
model = create_model(opt)
print('#audio segments = %d' % dataset_size)


# Forward pass
spectro_mag = []
spectro_pha = []
norm_params = []
audio = []
model.eval()
stride = opt.segment_length-opt.gen_overlap
with torch.no_grad():
    for i, data in enumerate(dataset):
        sr_spectro, sr_audio, lr_pha, norm_param, lr_spectro = model.inference(
            data['LR_audio'].cuda())
        print(sr_spectro.size())
        # spectro_mag.append(sr_spectro)
        # spectro_pha.append(lr_pha)
        # norm_params.append(norm_param)
        audio.append(sr_audio.cpu())

# Concatenate the audio
if opt.gen_overlap > 0:
    from torch.nn.functional import fold
    out_len = (dataset_size-1) * stride + opt.segment_length
    print(out_len)
    audio = torch.cat(audio,dim=0)
    print(audio.shape)
    audio[...,:opt.gen_overlap] *= 0.5
    audio[...,-opt.gen_overlap:] *= 0.5
    audio = audio.squeeze().transpose(-1,-2)
    audio = fold(audio, kernel_size=(1,opt.segment_length), stride=(1,stride), output_size=(1,out_len)).squeeze(0)
    audio = audio[...,opt.gen_overlap:-opt.gen_overlap]
    print(audio.shape)
else:
    audio = torch.cat(audio, dim=0).view(1, -1)
audio_len = data_loader.train_dataset.raw_audio.size(-1)
# print(audio.size())

# Evaluate the matrics
audio_len = data_loader.train_dataset.raw_audio.size(-1)
_mse, _snr_sr, _snr_lr, _ssnr_sr, _ssnr_lr, _pesq, _lsd = compute_matrics(
    data_loader.train_dataset.raw_audio, data_loader.train_dataset.lr_audio[..., :audio_len], audio[..., :audio_len], opt)
print('MSE: %.4f' % _mse)
print('SNR_SR: %.4f' % _snr_sr)
print('SNR_LR: %.4f' % _snr_lr)
#print('SSNR_SR: %.4f' % _ssnr_sr)
#print('SSNR_LR: %.4f' % _ssnr_lr)
#print('PESQ: %.4f' % _pesq)
print('LSD: %.4f' % _lsd)

# # Generate visuals
# lr_mag, _, sr_mag, _, _, _, _, _ = model.encode_input(
#     lr_audio=data_loader.dataset.lr_audio, hr_audio=audio)
# if opt.explicit_encoding:
#     lr_mag = 0.5*(lr_mag[:, 0, :, :]+lr_mag[:, 1, :, :])
#     sr_mag = 0.5*(sr_mag[:, 0, :, :]+sr_mag[:, 1, :, :])
# lr_spectro, lr_hist, _ = compute_visuals(
#     sp=lr_mag.squeeze().detach().cpu().numpy(), abs=True)
# sr_spectro, sr_hist, _ = compute_visuals(
#     sp=sr_mag.squeeze().detach().cpu().numpy(), abs=True)
# visuals = {'lable_spectro':         lr_spectro,
#            'generated_spectro':    sr_spectro,
#            'lable_hist':           lr_hist,
#            'generated_hist':       sr_hist}

# # Save files
# visualizer.display_current_results(visuals, 1, 1)
with open(os.path.join(opt.checkpoints_dir, opt.name, 'metric.txt'), 'w') as f:
    f.write('MSE,SNR_SR,LSD\n')
    f.write('%f,%f,%f' % (_mse, _snr_sr, _lsd))
sr_path = os.path.join(opt.checkpoints_dir, opt.name, 'sr_audio.wav')
lr_path = os.path.join(opt.checkpoints_dir, opt.name, 'lr_audio.wav')
hr_path = os.path.join(opt.checkpoints_dir, opt.name, 'hr_audio.wav')
torchaudio.save(sr_path, audio.cpu().to(torch.float32), opt.hr_sampling_rate)
torchaudio.save(lr_path, data_loader.train_dataset.lr_audio.cpu(),
                opt.hr_sampling_rate)
torchaudio.save(hr_path, data_loader.train_dataset.raw_audio.cpu(),
                data_loader.train_dataset.in_sampling_rate)
