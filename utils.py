from skimage import color, exposure
import numpy as np
import nibabel as nib
import torchvision.transforms.functional as TF
import random
from scipy import ndimage
import numpy as np
from sklearn.decomposition import PCA
import nibabel as nib
import cv2
from skimage import measure
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage import measure
from scipy.spatial import distance


# overlay a colored mask on image based on segmentation
def overlayMask(img, segmentation, RGB, intensity):
    """
    Description: This is a function that reture the image with a colored mask on it.
    img: gray value medical image
    segmentation: binary label image
    RGB: 'R', 'G', 'B'
    return: display the image

    For 2D images
    """

    alpha = intensity
    rows, cols = img.shape
    # Construct a colour image to superimpose
    color_mask = np.zeros((rows, cols, 3))
    # get the coordinates for the segmentation
    x_cords = []
    y_cords = []

    if RGB == 'R' :
        for i in range(rows):
            for j in range(cols):
                if segmentation[i, j] == 1:
                    color_mask[i, j] = [1, 0, 0] # Red block

    elif RGB == 'G' :
        for i in range(rows):
            for j in range(cols):
                if segmentation[i, j] == 1:
                    color_mask[i, j] = [0, 1, 0]  # Green block
    elif RGB == 'B' :
        for i in range(rows):
            for j in range(cols):
                if segmentation[i, j] == 1:
                    color_mask[i, j] = [0, 0, 1]  # Blue block

    # Construct RGB version of grey-level image
    img_color = np.dstack((img,img,img)).astype(np.uint8)
    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha
    img_masked = color.hsv2rgb(img_hsv)

    return img_masked

def contour_msk(img, seg, tgt_path, file_name):
    img = cv2.merge((img, img, img))
    # find the contour
    contours = measure.find_contours(seg, 0.5)
    if len(contours) == 2:
        if len(contours[0]) >= len(contours[1]):
            contours = contours[0]
        else:
            contours = contours[1]
    else:
        contours = contours[0]
    X = contours[:, 0]
    Y = contours[:, 1]
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.scatter(Y, X, s=0.003, c='r')
    plt.savefig(tgt_path + file_name + '.jpg')
    plt.close('all')

# Dataloader helps load the data within the minibatch
def Dataloader_1(data_path, name_list, indices, x_shape, y_shape, img_type, resize):
    """
    This is a function that help load the nii images into a minibatch with a set of indices as instruction
    data_path: the data path to load the data
    name_list: get the file names of images in the folder into a list
    indices: the corresponding indices used to form a minibatch of images
    return: return a minibatch of images

    For 2D images
    """
    # create zeros numpy arrays
    minibatch_imgs = np.zeros((x_shape, y_shape, indices.shape[0]))
    resize_imgs = np.zeros((319, 447, indices.shape[0]))

    if img_type == 'label':
        for i in range(indices.shape[0]):
            # store the images by the number in the array
            img = nib.load(data_path + name_list[indices[i]])
            img = img.get_data()
            img = np.rot90(np.array(img), k=1)
            img = img.squeeze()

            # threshold img
            img = np.where(img >= 0.5, 1, 0)
            img = CCP_processing(img)
            if indices[i] == 17 or indices[i] == 32 : # 17 & 32 22 & 40
                minibatch_imgs[:, :, i] = img[135:773, 378:1272]  # note: include or exclude
            else:
                minibatch_imgs[:, :, i] = img[80:718, 50:944]

            if resize == 'y':
                resize_imgs[:, :, i] = cv2.resize(minibatch_imgs[:, :, i], (447, 319), interpolation=cv2.INTER_AREA)

    else:
        for i in range(indices.shape[0]):
            # store the images by the number in the array
            img = nib.load(data_path + name_list[indices[i]])
            img = img.get_data()
            img = np.rot90(np.array(img), k = 1)
            img = img.squeeze()

            if indices[i] == 17 or indices[i] == 32: # 22 & 40
                minibatch_imgs[:, :, i] = img[135:773, 378:1272] # note: include or exclude
            else:
                minibatch_imgs[:, :, i] = img[80:718, 50:944]

            if resize == 'y':
                resize_imgs[:, :, i] = cv2.resize(minibatch_imgs[:, :, i], (447, 319), interpolation=cv2.INTER_AREA)

    if resize=='y':
        return resize_imgs
    else:
        return minibatch_imgs

# Dataloader helps load the data within the minibatch
def Dataloader_test_1(data_path, name_list, indices, img_type, resize):

    """
    This is a function that help load the nii images into a minibatch with a set of indices as instruction
    data_path: the data path to load the data
    name_list: get the file names of images in the folder into a list
    indices: the corresponding indices used to form a minibatch of images
    return: return a minibatch of images

    For 2D images
    """
    # create zeros numpy arrays
    img = nib.load(data_path + name_list[indices])
    img = img.get_data()
    img = np.rot90(np.array(img), k = 1)
    img = img.squeeze()

    if img_type == 'image':

        # normalising the images
        if indices == 17 or indices == 32:  # 22 & 40
            img = img[135:773, 378:1272]  # note: include or exclude
        else:
            img = img[80:718, 50:944]

        if resize == 'y':
            img = cv2.resize(img, (447,319), interpolation=cv2.INTER_AREA)

        img = (img-np.mean(img))/(np.std(img))

    else:
        img = np.where(img >= 0.5, 1, 0)
        img = CCP_processing(img)
        img = img.astype(float)
        if indices == 17 or indices == 32:  # 22 & 40
            img = img[135:773, 378:1272]  # note: include or exclude
        else:
            img = img[80:718, 50:944]

        if resize == 'y':
            img = cv2.resize(img, (447,319), interpolation=cv2.INTER_AREA)

        # img = np.where(img >= 0.5, 1, 0)
        # img = CCP_processing(img)

    return img

def Dataloader_2(data_path, name_list, indices, img_size, img_type, resize, tgt_size):

    # create zeros numpy arrays
    minibatch_imgs = np.zeros((img_size[0], img_size[1], indices.shape[0]))
    resize_imgs = np.zeros((tgt_size[0], tgt_size[1], indices.shape[0]))

    if img_type == 'label':
        for i in range(indices.shape[0]):
            # store the images by the number in the array
            img = nib.load(data_path + name_list[indices[i]])
            img = img.get_fdata()

            # threshold img
            img = np.where(img >= 0.5, 1, 0)
            img = CCP_processing(img)
            minibatch_imgs[:,:,i] = img

            if resize == 'y':
                resize_imgs[:, :, i] = cv2.resize(minibatch_imgs[:, :, i], (tgt_size[1], tgt_size[0]), interpolation=cv2.INTER_AREA)
                resize_imgs = np.where(resize_imgs >= 0.5, 1, 0)

    else:
        for i in range(indices.shape[0]):
            # store the images by the number in the array
            img = nib.load(data_path + name_list[indices[i]])
            img = img.get_fdata()

            minibatch_imgs[:, :, i] = img

            if resize == 'y':
                resize_imgs[:, :, i] = cv2.resize(minibatch_imgs[:, :, i], (tgt_size[1], tgt_size[0]), interpolation=cv2.INTER_AREA)

    if resize=='y':
        return resize_imgs
    else:
        return minibatch_imgs

def Dataloader_test_2(data_path, name_list, indices, img_type, resize, tgt_size):

    # create zeros numpy arrays
    img = nib.load(data_path + name_list[indices])
    img = img.get_fdata()

    if img_type == 'image':

        if resize == 'y':
            img = cv2.resize(img, (tgt_size[1], tgt_size[0]), interpolation=cv2.INTER_AREA)

    else:
        img = np.where(img >= 0.5, 1, 0)
        img = CCP_processing(img)
        img = img.astype(float)
        if resize == 'y':
            img = cv2.resize(img, (tgt_size[1], tgt_size[0]), interpolation=cv2.INTER_AREA)
            img = np.where(img >= 0.5, 1, 0)

    return img

def data_aug_rot(img,seg):

    # random rotation, scaling and deformation
    if random.random() > 0.5:
        angle = random.randint(-20, 20)
        img = TF.rotate(img, angle)
        seg = TF.rotate(seg, angle)
        # more transforms ...
        img = np.array(img)
        seg = np.array(seg)
    return img,seg

def data_aug_rot_single(img):

    # random rotation, scaling and deformation
    if random.random() > 0.5:
        angle = random.randint(-20, 20)
        img = TF.rotate(img, angle)
        # more transforms ...
        img = np.array(img)
    return img

def data_aug_gama(img):

    if random.random() > 0.5:
        img = np.uint8(img)
        gamma = random.uniform(0.5, 1.5)
        img = exposure.adjust_gamma(img, gamma)
    return img

def histo_trans(img):

    if random.random() > 0.5:
        img = np.uint8(img)
        clip = random.uniform(0.5,1.5)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        img = clahe.apply(img)
    return img


def norm(imgs):

    dim = imgs.shape[2]

    for i in range(0, dim):
        imgs[:, :, i] = (imgs[:, :, i] - np.mean(imgs[:, :, i]))/(np.std(imgs[ :, :, i]))

    return imgs

def CCP_processing(seg):

    new_img = np.zeros_like(seg)
    for val in np.unique(seg)[1:]:
        mask = np.uint8(seg == val)
        labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        new_img[labels == largest_label] = val

    return new_img


def computeQualityMeasures(lP, lT):
    labelPred = sitk.GetImageFromArray(lP, isVector=False)
    labelTrue = sitk.GetImageFromArray(lT, isVector=False)
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    Hausdorff = hausdorffcomputer.GetHausdorffDistance()
    return Hausdorff

def map_generator(seg):
    X, Y = np.nonzero(seg)
    dim = X.shape
    dim = int(dim[0])
    cord = np.zeros((dim, 2))
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    cord[:, 0] = X
    cord[:, 1] = Y
    pca = PCA(n_components=1)
    pca.fit(cord)
    X_pca = pca.transform(cord)
    X_axis = pca.inverse_transform(X_pca)
    X_new = np.zeros((dim,))
    Y_new = np.zeros((dim,))
    X_new = X_axis[:, 0]
    Y_new = X_axis[:, 1]
    # draw the end length
    slope = (np.max(X_new) - np.min(X_new))/(np.max(Y_new) - np.min(Y_new))
    slope = -1 / slope
    b1 = np.max(X_new) - slope * np.max(Y_new)
    b2 = np.min(X_new) - slope * np.min(Y_new)
    # plot the contour to check
    plt.figure()
    plt.imshow(seg, cmap='gray')
    plt.scatter(Y_new, X_new, s = 2)
    plot1 = range(int(np.max(Y_new)) - 30, int(np.max(Y_new)) + 30)
    plot2 = range(int(np.min(Y_new)) - 30, int(np.min(Y_new)) + 30)
    plt.plot(plot1, slope*plot1 + b1,color = 'y', ls = '--', linewidth = 3)
    plt.plot(plot2, slope*plot2 + b2, color = 'y', ls = '--', linewidth =3 )
    plt.show()

    return X_new

# this function is to check whether the binary label is set to 0 and 1
def bin_value_check(label):
    """
    This function is to check whether the pixels in binary label are 0 or 1.
    Label: binary image

    """
    place = label[np.where(np.logical_and(0 < label, label < 1))]
    if len(place) > 0:
        print('Y')

def dcm2nii(path_read, path_save, i, name_list):

    """
    This function is to transfer the dicom image to nifti file.
    path_read:  dcm file path
    path_save: nii file path
    name_list: the name list of the dcm files
    """
    name = []
    name.append(path_read + name_list[i])
    series_file_names = tuple(name)
    print(len(series_file_names))
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3d = series_reader.Execute()
    sitk.WriteImage(image3d, path_save + name_list[i].rstrip('dcm') + 'nii')

def len_measurement_all(seg, scaler):
    """
    Length measurement (PCA-all-points span):
    - Compute PCA on ALL positive pixels (seg==1) to get the 1D main axis.
    - Project and invert to get axis-aligned coordinates (X_new, Y_new).
    - Take the FULL span (max-min) along the PCA axis for all projected points
      WITHOUT restricting to the contour interior.
    - Convert pixel length to mm via `scaler`.

    Args:
        seg (ndarray, shape [H,W]): binary segmentation (0/1)
        scaler (float): pixel-to-mm scale factor

    Returns:
        float: length in mm
    """
    X, Y = np.nonzero(seg)
    dim = int(X.shape[0])
    cord = np.zeros((dim, 2))
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    cord[:, 0] = X; cord[:, 1] = Y

    pca = PCA(n_components=1)
    pca.fit(cord)
    X_pca = pca.transform(cord)
    X_axis = pca.inverse_transform(X_pca)

    X_new = X_axis[:, 0]
    Y_new = X_axis[:, 1]

    length_x = np.max(X_new) - np.min(X_new)
    length_y = np.max(Y_new) - np.min(Y_new)
    length = np.sqrt((length_x ** 2) + (length_y ** 2))
    return length * scaler


def len_measurement_VarPCA(seg, scaler, resize):
    """
    Length measurement (VarPCA – swept parallel axes inside contour):
    - Compute PCA main axis from positive pixels.
    - Build a family of lines PARALLEL to this PCA axis by shifting it up/down.
    - For each shifted line, compute the segment of projected points that lies
      INSIDE the polygonal contour, and take its Euclidean span.
    - Return the MAXIMUM span over all shifts, then convert to mm via `scaler`.

    Intuition:
    - If the organ is curved or the main axis is not centered, sweeping parallel
      axes can capture a longer internal chord than the single central axis.

    Args:
        seg (ndarray, shape [H,W]): binary segmentation (0/1)
        scaler (float): pixel-to-mm scale factor
        resize (str): 'n' for native ranges; otherwise smaller sweep step/range

    Returns:
        float: maximum swept length in mm
    """
    X, Y = np.nonzero(seg)
    dim = int(X.shape[0])
    cord = np.zeros((dim, 2))
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    cord[:, 0] = X; cord[:, 1] = Y

    pca = PCA(n_components=1)
    pca.fit(cord)
    X_pca = pca.transform(cord)
    X_axis = pca.inverse_transform(X_pca)
    X_new = X_axis[:, 0]
    Y_new = X_axis[:, 1]

    # choose the contour
    contours = measure.find_contours(seg, 0.5)
    if len(contours) == 2:
        contours = contours[0] if len(contours[0]) >= len(contours[1]) else contours[1]
    else:
        contours = contours[0]

    # line model parallel to PCA axis: X = slope * Y + beta
    slope_x = np.max(X_new) - np.min(X_new)
    slope_y = np.max(Y_new) - np.min(Y_new)
    slope = slope_x / (slope_y + 1e-8)
    beta = np.max(X_new) - slope * np.max(Y_new)

    Y_new = np.arange(1, 1112, 0.2)
    path = mpltPath.Path(contours)

    length_list = []
    var = np.arange(0, 80, 0.2) if resize == 'n' else np.arange(0, 40, 0.1)

    for i in var:
        # shift upward
        X_shift = Y_new * slope + beta + i
        inside = path.contains_points(np.asarray([X_shift, Y_new]).T)
        lx = np.max(X_shift[inside]) - np.min(X_shift[inside])
        ly = np.max(Y_new[inside]) - np.min(Y_new[inside])
        length_list.append(np.sqrt(lx**2 + ly**2))

        # shift downward
        X_shift = Y_new * slope + beta - i
        inside = path.contains_points(np.asarray([X_shift, Y_new]).T)
        lx = np.max(X_shift[inside]) - np.min(X_shift[inside])
        ly = np.max(Y_new[inside]) - np.min(Y_new[inside])
        length_list.append(np.sqrt(lx**2 + ly**2))

    return np.max(length_list) * scaler


def len_measurement_points(seg, scaler):
    """
    Length measurement (contour pairwise max distance):
    - Extract the polygonal contour of the segmentation.
    - Compute the pairwise Euclidean distances between ALL contour points.
    - The length is the MAXIMUM pairwise distance (the diameter along the contour),
      then multiplied by `scaler` to convert to mm.

    Notes:
    - This is a purely geometric diameter on the contour, independent of PCA.
    - Computationally heavier O(N^2) w.r.t number of contour points.

    Args:
        seg (ndarray, shape [H,W]): binary segmentation (0/1)
        scaler (float): pixel-to-mm scale factor

    Returns:
        float: length in mm
    """
    contours = measure.find_contours(seg, 0.5)
    contours = list(contours)
    if len(contours) == 2:
        if len(contours[0]) >= len(contours[1]):
            X = contours[0][:, 0]; Y = contours[0][:, 1]
        else:
            X = contours[1][:, 0]; Y = contours[1][:, 1]
    else:
        X = contours[0][:, 0]; Y = contours[0][:, 1]

    dim = int(X.shape[0])
    cord = np.zeros((dim, 2))
    cord[:, 0] = X
    cord[:, 1] = Y
    cord = [tuple(c) for c in cord]

    dists = distance.cdist(cord, cord, 'euclidean')
    max_dist_px = np.max(dists)
    return max_dist_px * scaler


