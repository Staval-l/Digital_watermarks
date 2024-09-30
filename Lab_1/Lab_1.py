import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_img_rgb(path_to_img: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path_to_img), cv2.COLOR_BGR2RGB)


def read_img_ycbcr(path_to_img: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path_to_img), cv2.COLOR_BGR2YCrCb)


def read_watermark(path_to_watermark: str) -> np.ndarray:
    return cv2.imread(path_to_watermark, cv2.IMREAD_UNCHANGED)


def plot_img(img):
    plt.imshow(img)
    plt.show()


def plot_RGB_channel(img):
    fig, axs = plt.subplots(nrows=1, ncols=4)
    axs[0].imshow(img)
    axs[1].imshow(img[:, :, 0], cmap='Reds')
    axs[2].imshow(img[:, :, 1], cmap='Greens')
    axs[3].imshow(img[:, :, 2], cmap='Blues')
    plt.show()


def plot_watermark(watermark):
    plt.imshow(watermark * 255, cmap='grey', vmin=0, vmax=255)
    plt.show()


def lsb_embled(C: np.ndarray, W: np.ndarray) -> np.ndarray:  # LSB for 1-4 var
    bit_img = np.unpackbits(C, bitorder='little').reshape((512, 512, 8))
    watermark = W
    watermark[watermark == 255] = 1
    bit_img[:, :, 0] = (bit_img[:, :, 0] ^ watermark)
    res = np.packbits(bit_img.reshape((512, 512 * 8)), bitorder='little').reshape((512, 512))
    return res


def lsb_disembled(C: np.ndarray, Cp: np.ndarray) -> np.ndarray:  # for 1-4 var
    bit_img = np.unpackbits(C, bitorder='little').reshape((512, 512, 8))
    bit_img_w = np.unpackbits(Cp, bitorder='little').reshape((512, 512, 8))
    watermark = bit_img_w[:, :, 0] ^ bit_img[:, :, 0]
    plot_img(watermark * 255)
    return watermark


def task_1(C: np.ndarray, W1: np.ndarray,
           W2: np.ndarray) -> np.ndarray:  # Green-2, Blue-1 ||| до, после изображение, битовые плоскости и ватермарка отдельно
    res_img = C.copy()
    bit_img_g = np.unpackbits(res_img[:, :, 1], bitorder='little').reshape((512, 512, 8))
    bit_img_b = np.unpackbits(res_img[:, :, 2], bitorder='little').reshape((512, 512, 8))
    watermark_1 = W1.copy()
    watermark_2 = W2.copy()
    watermark_1[watermark_1 == 255] = 1
    watermark_2[watermark_2 == 255] = 1
    bit_img_g[:, :, 1] = (bit_img_g[:, :, 1] ^ watermark_1)
    bit_img_b[:, :, 0] = (bit_img_b[:, :, 0] ^ watermark_2)
    res_g = np.packbits(bit_img_g.reshape((512, 512 * 8)), bitorder='little').reshape((512, 512))
    res_b = np.packbits(bit_img_b.reshape((512, 512 * 8)), bitorder='little').reshape((512, 512))
    res_img[:, :, 1] = res_g
    res_img[:, :, 2] = res_b
    # plot_img(res_img)
    return res_img


def task_2(C: np.ndarray, Cp: np.ndarray) -> list:
    orig_img = C.copy()
    watermarked_img = Cp.copy()
    bit_img_g = np.unpackbits(orig_img[:, :, 1], bitorder='little').reshape((512, 512, 8))
    bit_img_b = np.unpackbits(orig_img[:, :, 2], bitorder='little').reshape((512, 512, 8))
    bit_watermarked_img_g = np.unpackbits(watermarked_img[:, :, 1], bitorder='little').reshape((512, 512, 8))
    bit_watermarked_img_b = np.unpackbits(watermarked_img[:, :, 2], bitorder='little').reshape((512, 512, 8))
    watermark_1 = bit_watermarked_img_g[:, :, 1] ^ bit_img_g[:, :, 1]
    watermark_2 = bit_watermarked_img_b[:, :, 0] ^ bit_img_b[:, :, 0]
    plot_watermark(watermark_1)
    plot_watermark(watermark_2)
    return [watermark_1, watermark_2]


def calculate_delta(var_num: int):
    return 4 + 4 * (var_num % 3)


def calculate_teta(img: np.ndarray, var_num: int):  # (3.13)
    return img % var_num


def task_3(C: np.ndarray, watermark: np.ndarray) -> np.ndarray:
    res_img = C.copy()
    delta = calculate_delta(14)
    watermark[watermark == 255] = 1
    teta = calculate_teta(res_img[:, :, 2], 14)
    res_img[:, :, 2] = (np.floor(res_img[:, :, 2] / (2 * delta)) * (2 * delta) + watermark * delta + teta)
    # res_img = cv2.cvtColor(res_img, cv2.COLOR_YCrCb2RGB)
    return res_img


def task_4(C: np.ndarray, Cp: np.ndarray) -> np.ndarray:
    delta = calculate_delta(14)
    teta = calculate_teta(C[:, :, 2], 14)
    watermark = ((Cp[:, :, 2] - teta - (np.floor(C[:, :, 2] / (2 * delta)) * (2 * delta))) / delta)
    plot_watermark(watermark)
    return watermark


def main():
    path_to_img = 'Files/baboon.tif'
    paths_to_watermarks = ['Files/ornament.tif', 'Files/mickey.tif']
    var_num = 14
    orig_img_rgb = read_img_rgb(path_to_img)
    orig_img_ycbcr = read_img_ycbcr(path_to_img)
    # plot_img(orig_img_ycbcr)
    # plot_RGB_channel(orig_img_rgb)
    watermark_1 = read_watermark(paths_to_watermarks[0])
    watermark_2 = read_watermark(paths_to_watermarks[1])
    # Cp_1 = task_1(orig_img_rgb, watermark_1, watermark_2)
    # task_2(orig_img_rgb, Cp_1)
    Cp_2 = task_3(orig_img_ycbcr, watermark_2)
    plot_img(Cp_2)
    task_4(orig_img_ycbcr, Cp_2)


if __name__ == '__main__':
    main()
