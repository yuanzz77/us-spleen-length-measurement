import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats

import losses as loss
import utils
import unet_extrablock as nw

# ======================================
# CONFIG (mask private paths for GitHub)
# Replace these with your actual paths.
# ======================================
IMG_DIR     = './data/us_imgs/'
SEG_DIR     = './data/labels/'
INDEX_CSV   = './data/total_index.csv'           # split indices file (>= 420 rows)
MODEL_PATH  = './outputs/models/Unet/lr/3/unet.pt'  # path to trained weights
LENGTHS_CSV = './data/lengths.csv'               # per-index ground-truth spleen lengths (one value per index)

# Pixel → mm scaling factor (kept from your original script)
scaler = 0.18657135745244366 * 2

# --------------------------------------
# Load filenames (kept identical)
# --------------------------------------
imgs_namelist = sorted(os.listdir(IMG_DIR))
segs_namelist = sorted(os.listdir(SEG_DIR))

# --------------------------------------
# Load split indices (kept identical)
# --------------------------------------
total_index = np.uint(np.loadtxt(INDEX_CSV, delimiter=','))
training_index = total_index[0:252]
validation_index = total_index[252:336]
test_index = total_index[336:420]
del total_index

# --------------------------------------
# Load ground-truth length list
# (replaces hard-coded len_list/length_list)
# Keep the same variable name used later.
# --------------------------------------
# Expect LENGTHS_CSV to have at least 420 rows, one length per global index.
length_list = np.loadtxt(LENGTHS_CSV, delimiter=',').tolist()
test_list = []   # will store GT lengths for correlation

# --------------------------------------
# Prepare metrics containers
# --------------------------------------
losses = []
length_difference = []
pred_list = []   # for correlation scatter
distance = []    # for Hausdorff distances (in mm)

# --------------------------------------
# Initialize model & device (kept identical)
# --------------------------------------
unet = nw.unet()
unet.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet.to(device)
unet.eval()

# ======================================
# Inference over test split (unchanged)
# ======================================
for i in test_index:
    i = int(i)  # ensure python int for indexing

    # -------- Load one sample & place into canvas (384x576) --------
    imgs_tmp = utils.Dataloader_test_2(IMG_DIR, imgs_namelist, i, 'image', 'y', (382, 556))
    segs_tmp = utils.Dataloader_test_2(SEG_DIR, segs_namelist, i, 'label', 'y', (382, 556))
    test_segs = np.zeros((384, 576))
    test_segs[1:383, 10:566] = segs_tmp[:, :]
    test_imgs = np.zeros((384, 576))
    test_imgs[1:383, 10:566] = imgs_tmp[:, :]

    # Z-score normalization (kept)
    test_imgs = (test_imgs - np.mean(test_imgs)) / (np.std(test_imgs) + 1e-8)

    # -------- Ground-truth length for this index --------
    length_of_seg = length_list[i]
    test_list.append(length_of_seg)

    # -------- To tensors & move to device --------
    test_imgs_t = torch.from_numpy(test_imgs.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    test_segs_t = torch.from_numpy(test_segs.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    imgs_test = test_imgs_t.to(device, dtype=torch.float)
    segs_test = test_segs_t.to(device, dtype=torch.float)

    # -------- Forward inference --------
    with torch.no_grad():
        pred = unet.forward(imgs_test)

    # -------- Convert to numpy & binarize --------
    pred_test = pred.detach().cpu().numpy()
    test_segs_np = segs_test.detach().cpu().numpy()
    test_segs_np = np.squeeze(test_segs_np)
    pred_test = np.squeeze(pred_test)

    pred_test = np.where(pred_test >= 0.5, 1, 0).astype(np.uint8)

    # -------- Post-process (largest component etc.) --------
    pred_test = utils.CCP_processing(pred_test[:, :]).astype(np.uint8)

    # -------- Dice (after threshold) --------
    cost = loss.dice_test(pred_test, test_segs_np.astype(np.uint8))
    losses.append(cost)

    # -------- Length measurement (mm) --------
    length_of_pred = utils.len_measurement_all(pred_test, scaler)
    length_of_pred = round(length_of_pred, 1)
    pred_list.append(length_of_pred)

    # -------- Hausdorff distance (mm) via your utility --------
    haus_distance_px = utils.computeQualityMeasures(pred_test, test_segs_np.astype(np.uint8))
    haus_distance_mm = haus_distance_px * scaler
    distance.append(haus_distance_mm)

    # -------- Relative length error --------
    length_difference.append(abs(length_of_seg - length_of_pred) / (length_of_seg + 1e-8))

# --------------------------------------
# Aggregate metrics & print (kept)
# --------------------------------------
losses = np.array(losses)
dice = losses.mean()

length_difference = np.array(length_difference)
length_difference_mean = np.mean(length_difference)

distance = np.array(distance)
distance_mean = distance.mean()

print('\n\n\n\n')
print('Dice on each image:')
print(losses)
print('\n')
print('Mean dice over all test images:')
print(dice)
print('\n')
print('Mean hausdorff distance over all images (mm):')
print(distance_mean)
print('\n')
print('Mean length error percentage:')
print(length_difference_mean)

print('correlation')
r, p = stats.pearsonr(np.array(test_list), np.array(pred_list))
print(r)
print('\n')
print(p)
print(pred_list)
print('\n')
print('std of length error percentage')
print(np.std(length_difference))

# (Optional plotting code kept commented out, as in your original)
# fig = plt.figure()
# plt.scatter(length_list, pred_list)
# m,b = np.polyfit(length_list, pred_list, 1)
# plt.xlim((20, 185))
# plt.ylim((20, 185))
# plots = range(0, 180)
# plt.plot(plots, m*plots + b, color = 'y', label = 'Best Fit Line')
# plt.title('Correlation Scatter Plot')
# plt.xlabel('Ground Truth Lengths of Spleen')
# plt.ylabel('Predictions (Segmentation-based Approach)')
# plt.legend()
# plt.text(125, 50, 'correlation = ' + str(round(r, 3)))
# plt.savefig('./training_model/cv/results/u-net.png')
# plt.show()
