from skimage import io, exposure, filters, restoration
import numpy as np
import os
import re


def get_imgs_in_round(src_dir, round_id):
    f_list = [
        "".join([src_dir, f]) for f in os.listdir(src_dir)
        if f.startswith("".join(["R", str(round_id), "_"]))
        if re.search("_c1_", f) is None
    ]
    return f_list


def get_tissue_mask(f_list, sigma_thresh=1.0):
    '''
    :param f_list: list of filenames for images in round
    :param sigma_thresh: values greater than this sigma threshold indicate image is noisy, and excluded from mask
    :return: tissue mask, index of images used to create mask

    EXAMPLE:
    src_dir = "./images/"
    f_list = ["".join([src_dir, f]) for f in os.listdir(src_dir) if not f.startswith(".")]
    get_tissue_mask, keep_idx = get_tissue_mask(f_list)

    '''
    keep_idx = []
    tissue_mask = None
    for i, f in enumerate(f_list):

        img = io.imread(f, True)
        img = exposure.rescale_intensity(img, out_range=(0, 255))
        if tissue_mask is None:
            tissue_mask = np.zeros(img.shape, dtype=np.uint8)

        ### Exclude noisy images
        sigma_est = np.mean(
            restoration.estimate_sigma(img, multichannel=False))
        if sigma_est > sigma_thresh:
            print("skipping because too noisy", f)
            continue

        ### Use clean images to create mask
        keep_idx.append(i)
        max_val = np.percentile(img, 95)  ### Cap extreme values
        img[img > max_val] = max_val
        t = filters.threshold_otsu(img[img > 0])
        tissue_mask[img > t] += img[
            img >
            t]  #255 #+= mask # exposure.rescale_intensity(img, out_range=(0, 255))

    tissue_mask = exposure.rescale_intensity(tissue_mask, out_range=(0, 255))

    ### Threshold to get mask
    t = filters.threshold_otsu(tissue_mask[tissue_mask > 0])
    final_mask = np.zeros(tissue_mask.shape, dtype=np.uint8)
    final_mask[tissue_mask >= t] = 255
    eq_tissue_mask = exposure.equalize_hist(tissue_mask, mask=final_mask)
    eq_tissue_mask[final_mask == 0] = 0
    return eq_tissue_mask, keep_idx


from pathlib import Path

src_dir = "./NormalBreast/"
dst_dir = "./tissue_masks/"
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

for i in range(11):
    print("mask", i)
    round_f_list = get_imgs_in_round(src_dir, i)
    tissue_mask, keep_idx = get_tissue_mask(round_f_list, sigma_thresh=1.0)
    io.imsave("".join([dst_dir, str(i).zfill(2), "_tissue_mask.png"]),
              tissue_mask)
