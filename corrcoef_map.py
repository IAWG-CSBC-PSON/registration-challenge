import numpy as np
import matplotlib.pyplot as plt

from skimage import io, transform

def corrcoef_map(image1, image2, tile_shape):
    if image1.shape != image2.shape:
        raise ValueError('shape mismatch: image1 has shape {} and image2 has shape {}'\
                .format(image1.shape, image2.shape))
    num_dim = len(image1.shape)
    if num_dim != 2:
        raise NotImplementedError('image1 has shape {}, but currently only support 2D images'\
                .format(image1.shape))
    # grid construction
    x_grid = np.arange(image1.shape[0], dtype=int)
    y_grid = np.arange(image1.shape[1], dtype=int)
    factors = (int(image1.shape[0] / tile_shape[0]),
            int(image1.shape[1] / tile_shape[1]))
    cc = np.zeros_like(image1)
    for x_tile in np.array_split(x_grid, factors[0]):
        for y_tile in np.array_split(y_grid, factors[1]):
            xl, xu = x_tile.min(), x_tile.max()
            yl, yu = y_tile.min(), y_tile.max()
            tile1 = image1[xl:xu, yl:yu]
            tile2 = image2[xl:xu, yl:yu]
            cc[xl:xu, yl:yu] = np.corrcoef(tile1.flatten(), tile2.flatten())[0,1]
    return cc

def normalize(im):
    im = im.astype(float)
    im -= im.min()
    im /= im.max()
    return im

if __name__ == '__main__':
    # paths
    unreg_r1c1_path = 'R1_PCNA.CD8.PD1.CK19_SMT130-4_2019_05_08__12_15__1613S-Scene-012_c1_ORG.tif'
    unreg_r2c1_path = 'R2_CK5.HER2.ER.CD45_SMT130-4_2019_05_09__21_52__1654S-Scene-012_c1_ORG.tif'
    reg_r1c1_path = 'Registered-R1_PCNA.CD8.PD1.CK19_SMT130-4-Scene-012_c1_ORG.tif'
    reg_r2c1_path = 'Registered-R2_CK5.HER2.ER.CD45_SMT130-4-Scene-012_c1_ORG.tif'

    # load data and crop, as un-registered images have different sizes
    unreg_image1, unreg_image2 = io.imread(unreg_r1c1_path), io.imread(unreg_r2c1_path)
    unreg_image1, unreg_image2 = unreg_image1[0:5748, 0:5748], unreg_image2[0:5748, 0:5748]
    reg_image1, reg_image2 = io.imread(reg_r1c1_path), io.imread(reg_r2c1_path)

    # normalize and stack for RGB images
    unreg_image1, unreg_image2 = normalize(unreg_image1), normalize(unreg_image2)
    unreg_rgb = np.stack([unreg_image1, unreg_image2, np.zeros_like(unreg_image1)], axis=-1)

    reg_image1, reg_image2 = normalize(reg_image1), normalize(reg_image2)
    reg_rgb = np.stack([reg_image1, reg_image2, np.zeros_like(reg_image1)], axis=-1)

    # calculate tile-based correlation coefficient
    width = 100
    unreg_cc = corrcoef_map(unreg_image1, unreg_image2, tile_shape=(width, width))
    reg_cc = corrcoef_map(reg_image1, reg_image2, tile_shape=(width, width))

    # plotting
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(6,6))
    p = dict(fraction=0.046, pad=0.04)

    plt.subplot(221)
    plt.imshow(unreg_rgb)
    plt.title('R1c1 (red), R2c1 (green)')
    plt.xticks([]); plt.yticks([])

    plt.subplot(222)
    plt.imshow(unreg_cc, cmap='coolwarm')
    plt.title('corrcoef map')
    plt.colorbar(**p)
    plt.xticks([]); plt.yticks([])

    plt.subplot(223)
    plt.imshow(reg_rgb)
    plt.title('Registered version')
    plt.xticks([]); plt.yticks([])

    plt.subplot(224)
    plt.imshow(reg_cc, cmap='coolwarm')
    plt.title('corrcoef map')
    plt.colorbar(**p)
    plt.xticks([]); plt.yticks([])

    plt.tight_layout()
    plt.savefig('example.png')
    plt.show()
