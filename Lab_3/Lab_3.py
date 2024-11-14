import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import convolve2d
import pandas as pd


def read_img(path_to_img) -> np.ndarray:
    return cv2.imread(path_to_img, cv2.IMREAD_GRAYSCALE)


def PSNR(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = (np.sum((img1 - img2) ** 2)) / (512 * 512)
    return 10 * np.log10(255 * 255 / mse)


def watermark_extraction(C: np.ndarray, Cw: np.ndarray, alph: float):
    f = np.abs(np.fft.fftshift(np.fft.fft2(C)))
    fw = np.abs(np.fft.fftshift(np.fft.fft2(Cw)))
    w = (fw - f) / (alph * f)
    return w


def calc_po(C: np.ndarray, Cw: np.ndarray, omega: np.ndarray, alph: float):
    omega_w = watermark_extraction(C, Cw, alph)
    multi = omega * omega_w
    sum = np.sum(multi)
    sqrt1 = np.sqrt(np.sum(omega ** 2))
    sqrt2 = np.sqrt(np.sum(omega_w ** 2))
    return sum / (
            sqrt1 *
            sqrt2
    )


def cut_method(C: np.ndarray, Cw: np.ndarray, p: float) -> np.ndarray:
    result = C.copy()
    new_sides = [round(C.shape[0] * math.sqrt(p)), round(C.shape[1] * math.sqrt(p))]
    result[0:new_sides[0], 0:new_sides[1]] = Cw[0:new_sides[0], 0:new_sides[1]]
    return result


def scale_method(Cw: np.ndarray, p: float) -> np.ndarray:
    new_sides = [round(Cw.shape[0] * p), round(Cw.shape[1] * p)]
    result = np.zeros_like(Cw)
    resized_image = cv2.resize(Cw, (new_sides[0], new_sides[1]))
    if p <= 1.0:
        result[:new_sides[0], :new_sides[1]] = resized_image
    else:
        result = resized_image[:result.shape[0], :result.shape[1]]
    return result


def gauss_blur(Cw: np.ndarray, sigma: float) -> np.ndarray:
    m = 2 * math.floor(3 * sigma) + 1  # window size
    window = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            window[i][j] = np.exp(-((((i - m / 2) ** 2) + ((j - m / 2) ** 2)) / (2 * sigma ** 2)))
    k = 1 / np.sqrt(window)
    g = k * window
    result = convolve2d(Cw, g, mode='same')
    return result


def wh_noise(Cw: np.ndarray, disp: float) -> np.ndarray:
    noise = np.random.normal(0, np.sqrt(disp), size=Cw.shape)
    result = Cw + noise
    return result


def task_1(C: np.ndarray, Cw: np.ndarray, omega: np.ndarray, alph: float, orig_po: float) -> float:
    p_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    po_list, PSNR_list = [], []
    for p in p_list:
        new_Cw = cut_method(C, Cw, p)
        po_list.append(calc_po(C, new_Cw, omega, alph))
        PSNR_list.append(PSNR(C, new_Cw))
    # plt.plot(np.arange(0, len(po_list), 1), po_list, marker="o", linestyle='-',
    #          c="red")
    plt.plot(p_list, po_list, marker="o", linestyle='-',
             c="red")
    plt.axhline(orig_po, color='blue', linestyle='--', linewidth=2, label='a')
    plt.title('Cut method\nRho value')
    plt.show()
    # plt.plot(p_list, PSNR_list, marker="o", linestyle='-',
    #          c="red")
    # plt.title('Cut method\nPSNR value')
    # plt.show()
    return np.mean(po_list)


def task_2(C: np.ndarray, Cw: np.ndarray, omega: np.ndarray, alph: float, orig_po: float) -> float:
    p_list = [0.55, 0.7, 0.85, 1.0, 1.15, 1.30, 1.45]
    po_list, PSNR_list = [], []
    for p in p_list:
        new_Cw = scale_method(Cw, p)
        po_list.append(calc_po(C, new_Cw, omega, alph))
        PSNR_list.append(PSNR(C, new_Cw))
    plt.plot(p_list, po_list, marker="o", linestyle='-',
             c="red")
    plt.axhline(orig_po, color='blue', linestyle='--', linewidth=2, label='a')
    plt.title('Scale method\nRho value')
    plt.show()
    # plt.plot(p_list, PSNR_list, marker="o", linestyle='-',
    #          c="red")
    # plt.title('Scale method\nPSNR value')
    # plt.show()
    return np.mean(po_list)


def task_3(C: np.ndarray, Cw: np.ndarray, omega: np.ndarray, alph: float, orig_po: float) -> float:
    p_list = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    po_list, PSNR_list = [], []
    for p in p_list:
        new_Cw = gauss_blur(Cw, p)
        po_list.append(calc_po(C, new_Cw, omega, alph))
        PSNR_list.append(PSNR(C, new_Cw))
    plt.plot(p_list, po_list, marker="o", linestyle='-',
             c="red")
    plt.axhline(orig_po, color='blue', linestyle='--', linewidth=2, label='a')
    plt.title('Gauss blur method\nRho value')
    plt.show()
    # plt.plot(p_list, PSNR_list, marker="o", linestyle='-',
    #          c="red")
    # plt.title('Gauss blur method\nPSNR value')
    # plt.show()
    return np.mean(po_list)


def task_4(C: np.ndarray, Cw: np.ndarray, omega: np.ndarray, alph: float, orig_po: float) -> float:
    p_list = [400, 500, 600, 700, 800, 900, 1000]
    po_list, PSNR_list = [], []
    for p in p_list:
        new_Cw = wh_noise(Cw, p)
        po_list.append(calc_po(C, new_Cw, omega, alph))
        PSNR_list.append(PSNR(C, new_Cw))
    plt.plot(p_list, po_list, marker="o", linestyle='-',
             c="red")
    plt.axhline(orig_po, color='blue', linestyle='--', linewidth=2, label='a')
    plt.title('White noise method\nRho value')
    plt.show()
    # plt.plot(p_list, PSNR_list, marker="o", linestyle='-',
    #          c="red")
    # plt.title('White noise method\nPSNR value')
    # plt.show()
    return np.mean(po_list)


def task_5(C: np.ndarray, Cw: np.ndarray, omega: np.ndarray, alph: float, orig_po: float):
    p_list_1 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    p_list_2 = [0.55, 0.7, 0.85, 1.0, 1.15, 1.30, 1.45]
    results = []
    for p1 in p_list_1:
        for p2 in p_list_2:
            new_Cw = cut_method(C, Cw, p1)
            res = gauss_blur(new_Cw, p2)
            po = calc_po(C, res, omega, alph)
            psnr = PSNR(C, res)
            results.append([p1, p2, po, psnr])
    return results


def main():
    # a = 0.24900000000000014
    # Watermark 1: p = 0.5000953638847201, PSNR = 32.64433745043188
    # orig_po = 0.5000953638847201
    np.random.seed(666)
    a = 0.24900000000000014
    p_mean = []
    path_to_img = r'C:\Users\vodnyy\PycharmProjects\watermarking\lab2\bridge.tif'
    path_to_w_img = r'lab_3_watermarked.png'
    C = read_img(path_to_img)
    Cw = read_img(path_to_w_img)
    # omega = watermark_extraction(C, Cw, a)
    omega = read_img(r"watermark.png")
    orig_po = calc_po(C, Cw, omega, a)
    print("Original po =", orig_po)
    p_mean.append(task_1(C, Cw, omega, a, orig_po))
    p_mean.append(task_2(C, Cw, omega, a, orig_po))
    p_mean.append(task_3(C, Cw, omega, a, orig_po))
    p_mean.append(task_4(C, Cw, omega, a, orig_po))
    print("Mean po cut method =", p_mean[0])
    print("Mean po scale method =", p_mean[1])
    print("Mean po Gauss blur method =", p_mean[2])
    print("Mean po White noise method =", p_mean[3])
    res = pd.DataFrame(np.array(task_5(C, Cw, omega, a, orig_po)), columns=['p1', 'p2', 'po', 'psnr'])
    print("Mean po cut+scale =", res['po'].mean())
    print("Mean PSNR cut+scale =", res['psnr'].mean())
    print(res)


if __name__ == '__main__':
    main()
