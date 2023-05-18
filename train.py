from data.data_loader import CreateDataLoader
import signal
from util.util import compute_matrics
from util.visualizer import Visualizer
from options.train_options import TrainOptions
from models.models import create_model

import math
import os
import time
import csv
import gc

import numpy as np
import torch



def lcm(a, b): return abs(a * b)/math.gcd(a, b) if a and b else 0


# import debugpy
# debugpy.listen(("localhost", 5678))
# debugpy.wait_for_client()
# os.environ['CUDA_VISIBLE_DEVICES']='0'
torch.backends.cudnn.benchmark = True
# Get the training options
opt = TrainOptions().parse()
# Set the seed
torch.manual_seed(opt.seed)
# Set the path for save the trainning losses
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
eval_path = os.path.join(opt.checkpoints_dir, opt.name, 'eval.csv')

if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(
            iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

# Create the data loader
data_loader = CreateDataLoader(opt)
train_dataloader = data_loader.get_train_dataloader()
train_dataset_size = len(data_loader)
eval_dataloader = data_loader.get_eval_dataloader()
eval_dataset_size = data_loader.eval_data_len()
print('#training data = %d' % train_dataset_size)
print('#evaluating data = %d' % eval_dataset_size)

# Create the model
model = create_model(opt)
visualizer = Visualizer(opt)
optimizer_G, optimizer_D = model.optimizer_G, model.optimizer_D

# IMDCT for evaluation
# from util.util import kbdwin, imdct
# # from dct.dct import IDCT
# # _idct = IDCT()
# _imdct = IMDCT4(window=kbdwin, win_length=opt.win_length, hop_length=opt.hop_length, n_fft=opt.n_fft, center=opt.center, out_length=opt.segment_length, device = 'cuda')

if opt.fp16:
    from torch.cuda.amp import autocast as autocast
    from torch.cuda.amp import GradScaler
    # According to the offical tutorial, use only one GradScaler and backward losses separately
    # https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-multiple-models-losses-and-optimizers
    scaler = GradScaler()


# Set frequency for displaying information and saving
opt.print_freq = lcm(opt.print_freq, opt.batchSize)
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10
total_steps = (start_epoch-1) * train_dataset_size + epoch_iter
display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq
eval_delta = total_steps % opt.eval_freq if opt.validation_split > 0 else -1
# loss_update_delta = total_steps % opt.loss_update_freq if opt.use_time_D or opt.use_match_loss else -1

# Safe ctrl-c
end = False


def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    global end
    end = True


signal.signal(signal.SIGINT, signal_handler)

# Evaluation process
# Wrap it as a function so that I dont have to free up memory manually


def eval_model():
    err = []
    snr = []
    snr_seg = []
    pesq = []
    lsd = []
    for j, eval_data in enumerate(eval_dataloader):
        model.eval()
        lr_audio = eval_data['LR_audio'].cuda()
        hr_audio = eval_data['HR_audio'].cuda()
        with torch.no_grad():
            _, sr_audio, _, _, _ = model.inference(lr_audio)
            _mse, _snr_sr, _snr_lr, _ssnr_sr, _ssnr_lr, _pesq, _lsd = compute_matrics(
                hr_audio.squeeze(), lr_audio.squeeze(), sr_audio.squeeze(), opt)
            err.append(_mse)
            snr.append((_snr_lr, _snr_sr))
            snr_seg.append((_ssnr_lr, _ssnr_sr))
            pesq.append(_pesq)
            lsd.append(_lsd)
        if j >= opt.eval_size:
            break

    eval_result = {'err': np.mean(err), 'snr': np.mean(snr), 'snr_seg': np.mean(
        snr_seg), 'pesq': np.mean(pesq), 'lsd': np.mean(lsd)}
    with open(eval_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=eval_result.keys())
        if csv_file.tell() == 0:
            writer.writeheader()
        writer.writerow(eval_result)
    print('Evaluation:', eval_result)
    model.train()


# Training...
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % train_dataset_size
    if epoch > opt.niter_limit_aux:
        model.limit_aux_loss = True
    for i, data in enumerate(train_dataloader, start=epoch_iter):
        if end:
            print('exiting and saving the model at the epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
            exit(0)
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # Whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        if opt.fp16:
            with autocast():
                losses, _ = model._forward(
                    data['LR_audio'].cuda(), data['HR_audio'].cuda(), infer=False)
        else:
            losses, _ = model._forward(
                data['LR_audio'].cuda(), data['HR_audio'].cuda(), infer=False)

        # Sum per device losses
        losses = [torch.mean(x) if not isinstance(x, int)
                  else x for x in losses]
        loss_dict = dict(zip(model.loss_names, losses))

        # Calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5 + (loss_dict.get('D_fake_t', 0) + loss_dict.get(
            'D_real_t', 0))*0.5 + (loss_dict.get('D_fake_mr', 0) + loss_dict.get('D_real_mr', 0))*0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_mat', 0) + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get(
            'G_VGG', 0) + loss_dict.get('G_GAN_t', 0) + loss_dict.get('G_GAN_mr', 0) + loss_dict.get('G_shift', 0)

        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()
        if opt.fp16:
            #with amp.scale_loss(loss_G, optimizer_G) as scaled_loss: scaled_loss.backward()
            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            # update the scaler only once per iteration
            # scaler.update()
        else:
            loss_G.backward()
            optimizer_G.step()

        # update discriminator weights
        optimizer_D.zero_grad()
        if opt.fp16:
            #with amp.scale_loss(loss_D, optimizer_D) as scaled_loss: scaled_loss.backward()
            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)
            scaler.update()
        else:
            loss_D.backward()
            optimizer_D.step()

        ############## Display results and errors ##########
        # print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(
                v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

        # display output images
        if save_fake:
            visuals = model.get_current_visuals()
            visualizer.display_current_results(visuals, epoch, total_steps)
            del visuals

        # save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if total_steps % opt.eval_freq == eval_delta:
            del losses, loss_D, loss_G, loss_dict
            torch.cuda.empty_cache()
            gc.collect()
            eval_model()
            torch.cuda.empty_cache()
            gc.collect()
        # if total_steps % opt.loss_update_freq == loss_update_delta:
        #     if opt.use_match_loss:
        #         model.update_match_loss_scaler()
        #     if opt.use_time_D:
        #         model.update_time_D_loss_scaler()

        if epoch_iter >= train_dataset_size:
            break

    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    # save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    # instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.update_fixed_params()

    # linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.update_learning_rate()
