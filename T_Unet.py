import os
import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

import losses as loss
import utils
import unet_extrablock as nw

# =========================
# PATH CONFIG (masked for GitHub)
# Replace with your local paths when running.
# =========================
img_path = './data/us_imgs/'                  # path to ultrasound images
seg_path = './data/labels/'                   # path to segmentation masks
index_csv = './data/total_index.csv'          # CSV file with train/val/test indices
save_root = './outputs/models/Unet/wd/'       # base directory to save models & curves

# =========================
# TRAINING HYPERPARAMS
# =========================
iterations = 63000                            # total training iterations
minibatch_size = 4                            # minibatch size
number_of_minibatch = int(252 / minibatch_size)  # minibatches per epoch (kept as original)
DC = loss.dice                                # dice loss function
list_wd = [1e-5, 1e-6, 1e-7, 1e-8]            # weight decays to try
save_index = 4                                 # starting index for save subfolders

# =========================
# PREPARE FILE LISTS
# =========================
imgs_namelist = sorted(os.listdir(img_path))
segs_namelist = sorted(os.listdir(seg_path))

# Split indices: 0:252 train, 252:336 val, 336:420 test (unchanged)
total_index = np.uint(np.loadtxt(index_csv, delimiter=','))
training_index = total_index[0:252]
validation_index = total_index[252:336]
test_index = total_index[336:420]
del total_index

# =========================
# TRAINING OVER MULTIPLE WDs
# =========================
for wd in list_wd:
    # Initialize model & optimizer (unchanged)
    unet = nw.unet()
    opt = torch.optim.Adam(unet.parameters(), lr=1e-3, weight_decay=wd)

    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet.to(device)

    # Bookkeeping
    counter = 0
    training_losses = []
    validation_losses = []
    epoch_training_error = []
    epoch_validation_error = []
    error_compare = 100
    smallest_epoch = 0
    mean_error_after_this_epoch = 0
    save_index += 1

    # Make save dir (mirrors your original per-wd subfolders)
    save_dir = os.path.join(save_root, f'{save_index}')
    os.makedirs(save_dir, exist_ok=True)

    # =========================
    # TRAINING LOOP
    # =========================
    for step_training in range(iterations):

        # Model to train mode; zero grads
        unet.train()
        opt.zero_grad()

        # Shuffle training indices at the start of each epoch
        if step_training in range(0, iterations, number_of_minibatch):
            random.shuffle(training_index)

        # Select current minibatch indices
        idx_of_minibatch = int(step_training % number_of_minibatch)
        idx_training = training_index[idx_of_minibatch*minibatch_size:(idx_of_minibatch+1)*minibatch_size]

        # -------- Data loading (kept exactly) --------
        # Load cropped content via your custom loader, then place onto fixed canvas
        imgs_tmp = utils.Dataloader_2(img_path, imgs_namelist, idx_training, (764,1112), 'image', 'y', (382,556))
        segs_tmp = utils.Dataloader_2(seg_path, segs_namelist, idx_training, (764,1112), 'image', 'y', (382,556))

        segs = np.zeros((384,576,4))
        imgs = np.zeros((384,576,4))
        for tmp in range(4):
            segs[1:383,10:566,tmp] = segs_tmp[:,:,tmp]
            imgs[1:383,10:566,tmp] = imgs_tmp[:,:,tmp]
        del tmp, segs_tmp, imgs_tmp

        # -------- Data augmentation (order preserved) --------
        for i in range(minibatch_size):
            imgs[:, :, i] = utils.data_aug_gama(imgs[:, :, i])
            imgs[:, :, i] = utils.histo_trans(imgs[:, :, i])
            imgs[:, :, i], segs[:, :, i] = utils.data_aug_rot(
                Image.fromarray(imgs[:, :, i]),
                Image.fromarray(segs[:, :, i])
            )

        # -------- Normalization & tensor conversion (kept) --------
        imgs = utils.norm(imgs)
        imgs = torch.from_numpy(imgs.astype(np.float32))
        segs = torch.from_numpy(segs.astype(np.float32))
        imgs = imgs.unsqueeze(0).permute(3, 0, 1, 2)   # (B, C, H, W) with C=1
        segs = segs.unsqueeze(0).permute(3, 0, 1, 2)

        imgs_training = imgs.to(device, dtype=torch.float)
        segs_training = segs.to(device, dtype=torch.float)

        # Forward + loss + backward (unchanged)
        pred = unet.forward(imgs_training)
        cost = DC(pred, segs_training)
        cost.backward()
        opt.step()

        # Record minibatch loss
        epoch_training_error.append(cost.item())

        # Clear interm. tensors and free CUDA cache
        del imgs_training, segs_training, pred, cost
        torch.cuda.empty_cache()

        # =========================
        # END OF EPOCH → VALIDATION
        # =========================
        if (step_training + 1) % number_of_minibatch == 0:
            counter += 1
            training_time = time.asctime(time.gmtime())
            print('Current time: %s' % training_time)
            print('Current mean loss over ' + str(counter) + ' epochs:  ' +
                  str(np.mean(np.array(epoch_training_error))))
            training_losses.append(np.mean(np.array(epoch_training_error)))
            epoch_training_error = []

            # Switch to eval
            opt.zero_grad()
            unet.eval()

            # -------- Validation over test split (kept) --------
            for j in test_index:
                # Load single sample, place onto canvas, z-score normalize
                imgs_tmp = utils.Dataloader_test_2(img_path, imgs_namelist, j, 'image', 'y', (382,556))
                segs_tmp = utils.Dataloader_test_2(seg_path, segs_namelist, j, 'label', 'y', (382,556))
                segs_val = np.zeros((384,576))
                imgs_val = np.zeros((384,576))
                segs_val[1:383,10:566] = segs_tmp[:,:]
                imgs_val[1:383,10:566] = imgs_tmp[:,:]
                del segs_tmp, imgs_tmp

                imgs_val = (imgs_val - np.mean(imgs_val)) / (np.std(imgs_val) + 1e-8)

                # To tensors (1,1,H,W) and compute loss
                imgs_t = torch.from_numpy(imgs_val.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                segs_t = torch.from_numpy(segs_val.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                pred = unet.forward(imgs_t)
                cost = DC(pred, segs_t)
                epoch_validation_error.append(cost.item())

            # Clear memory post validation
            del imgs_t, segs_t, pred, cost
            torch.cuda.empty_cache()

            # Epoch-level validation summary & best tracking
            print('Current testing error after ' + str(counter) + ' epochs:  ' +
                  str(np.mean(np.array(epoch_validation_error))))
            mean_error_after_this_epoch = np.mean(np.array(epoch_validation_error))
            validation_losses.append(mean_error_after_this_epoch)
            n_epochs = (step_training + 1) / (number_of_minibatch)

            # Save best model (path masked; logic unchanged)
            if mean_error_after_this_epoch < error_compare:
                error_compare = mean_error_after_this_epoch
                smallest_epoch = n_epochs
                torch.save(unet.state_dict(), os.path.join(save_dir, 'unet.pt'))

            X = range(1, int(n_epochs) + 1)
            print('Note: the smallest error ' + str(error_compare) +
                  ' is after ' + str(int(smallest_epoch)) + ' at the moment.')
            print('\n')

            # Plot and save loss curves (path masked; logic unchanged)
            plt.figure()
            plt.plot(X, training_losses, color='b', label='Training Error over Current Epoch')
            plt.plot(X, validation_losses, color='r', label='Testing Error after Current Epoch')
            plt.xlabel('Number of Epochs')
            plt.ylabel('Error')
            plt.legend()
            plt.savefig(os.path.join(save_dir, 'losscurve.png'))
            plt.close('all')

            # Reset validation accumulator for next epoch
            epoch_validation_error = []
