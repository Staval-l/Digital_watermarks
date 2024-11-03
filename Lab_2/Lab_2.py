import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath


def read_img(path_to_img) -> np.ndarray:
    return cv2.imread(path_to_img, cv2.IMREAD_GRAYSCALE)


def task_1(img_shape) -> np.ndarray:  # task_1
    watermark = np.zeros((4, img_shape[0], img_shape[1]))
    random_seq = np.random.normal(loc=1, scale=2, size=(
        (int(math.sqrt((img_shape[0] * img_shape[1]) / 16)), int(math.sqrt((img_shape[0] * img_shape[1]) / 16)))))
    watermark[0][(img_shape[0] // 4): (img_shape[0] // 4 + img_shape[0] // 4),
    (img_shape[1] // 4): (img_shape[1] // 4 + img_shape[1] // 4)] += random_seq
    watermark[1][(img_shape[0] // 4): (img_shape[0] // 4 + img_shape[0] // 4),
    (2 * img_shape[1] // 4): (2 * img_shape[1] // 4 + img_shape[1] // 4)] += random_seq
    watermark[2][(2 * img_shape[0] // 4): (2 * img_shape[0] // 4 + img_shape[0] // 4),
    (img_shape[1] // 4): (img_shape[1] // 4 + img_shape[1] // 4)] += random_seq
    watermark[3][(2 * img_shape[0] // 4): (2 * img_shape[0] // 4 + img_shape[0] // 4),
    (2 * img_shape[1] // 4): (2 * img_shape[1] // 4 + img_shape[1] // 4)] += random_seq
    fig, axs = plt.subplots(nrows=1, ncols=4)
    axs[0].imshow(np.abs(watermark[0]), cmap='gray')
    axs[1].imshow(np.abs(watermark[1]), cmap='gray')
    axs[2].imshow(np.abs(watermark[2]), cmap='gray')
    axs[3].imshow(np.abs(watermark[3]), cmap='gray')
    plt.show()
    return np.abs(watermark)


def task_2(C: np.ndarray) -> np.ndarray:  # task_2
    f = np.fft.fft2(C)
    return np.fft.fftshift(f)


def task_3(f: np.ndarray, omega: np.ndarray, alph: float) -> np.ndarray:  # task_3
    vector_polar = np.vectorize(cmath.polar)
    f_abs, f_phase = vector_polar(f)
    fw_abs = f_abs * (1 + (alph * omega))
    vector_rect = np.vectorize(cmath.rect)
    fw = vector_rect(fw_abs, f_phase)
    return fw


def task_4(fw: np.ndarray, write_path: str) -> None:
    res = np.fft.ifftshift(fw)
    res = np.fft.ifft2(res)
    res = np.abs(res)
    cv2.imwrite(write_path, res)


def task_5(path: str) -> np.ndarray:
    Cw = read_img(path)
    fw = task_2(Cw)
    return fw


def task_6(f: np.ndarray, fw: np.ndarray, omega: np.ndarray, alph: float):
    omega_w = (np.abs(fw) - np.abs(f)) / (alph * np.abs(f))
    multi = omega * omega_w
    sum = np.sum(multi)
    sqrt1 = np.sqrt(np.sum(omega ** 2))
    sqrt2 = np.sqrt(np.sum(omega_w ** 2))
    return sum / (
            sqrt1 *
            sqrt2
    )


def PSNR(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = (np.sum((img1 - img2) ** 2)) / (512 * 512)
    return 10 * np.log10(255 * 255 / mse)


def task_7(img, omega, write_path):
    alphs = []
    ps = []
    psnr_val = []
    alph = 0.1
    while alph <= 1:
        alphs.append(alph)
        task_4((task_3(task_2(img), omega, alph)), write_path)
        ps.append(task_6(task_2(img), task_5(write_path), omega, alph))
        psnr_val.append(PSNR(img, read_img(write_path)))
        alph += 0.1
    plt.plot(alphs, ps)
    plt.show()
    porog = alphs[ps.index(max(ps))]
    alph = porog - 0.1
    alphs.clear()
    ps.clear()
    psnr_val.clear()
    while alph <= porog + 0.2:
        alphs.append(alph)
        task_4((task_3(task_2(img), omega, alph)), write_path)
        ps.append(task_6(task_2(img), task_5(write_path), omega, alph))
        psnr_val.append(PSNR(img, read_img(write_path)))
        alph += 0.001
    plt.plot(alphs, ps)
    plt.show()
    print("p =", max(ps))
    best_a = alphs[ps.index(max(ps))]
    print("a =", alphs[ps.index(max(ps))])
    return best_a


def task_8(C: np.ndarray, watermarks, alpha: float):
    task_4((task_3(task_2(C), watermarks[1], alpha)), "watermarked_2.png")
    res_1 = [task_6(task_2(C), task_5("watermarked_2.png"), watermarks[1], alpha),
             PSNR(C, read_img("watermarked_2.png"))]
    task_4((task_3(task_2(C), watermarks[2], alpha)), "watermarked_3.png")
    res_2 = [task_6(task_2(C), task_5("watermarked_3.png"), watermarks[2], alpha),
             PSNR(C, read_img("watermarked_3.png"))]
    task_4((task_3(task_2(C), watermarks[3], alpha)), "watermarked_4.png")
    res_3 = [task_6(task_2(C), task_5("watermarked_4.png"), watermarks[3], alpha),
             PSNR(C, read_img("watermarked_4.png"))]
    print(f"Watermark 2: p = {res_1[0]}, PSNR = {res_1[1]}")
    print(f"Watermark 3: p = {res_2[0]}, PSNR = {res_2[1]}")
    print(f"Watermark 4: p = {res_3[0]}, PSNR = {res_3[1]}")
    return [res_1, res_2, res_3]


def plot_imges(C: np.ndarray):
    plt.imshow(C, cmap='gray')
    plt.title('Original Image')
    plt.show()
    plt.imshow(read_img("watermarked_1.png"), cmap='gray')
    plt.title('Watermarked 1')
    plt.show()
    plt.imshow(read_img("watermarked_2.png"), cmap='gray')
    plt.title('Watermarked 2')
    plt.show()
    plt.imshow(read_img("watermarked_3.png"), cmap='gray')
    plt.title('Watermarked 3')
    plt.show()
    plt.imshow(read_img("watermarked_4.png"), cmap='gray')
    plt.title('Watermarked 4')
    plt.show()


def main():
    path_to_img = r'C:\Users\Staval\PycharmProjects\Digital_watermarks\Lab_2\Files\goldhill.tif'
    write_path = 'watermarked_1.png'
    C = read_img(path_to_img)
    plt.imshow(np.log(np.abs(task_2(C))), cmap='gray')
    plt.title('Спектр изображения до встраивания')
    plt.show()
    watermarks = task_1(C.shape)
    omega = watermarks[0]
    alph = task_7(C, omega, write_path)
    plt.imshow(np.log(np.abs(task_3(task_2(C), watermarks[0], alph))), cmap='gray')
    plt.title('Спектр изображения после встраивания')
    plt.show()
    task_4(task_3(task_2(C), omega, alph), write_path)
    print(
        f"Watermark 1: p = {task_6(task_2(C), task_5(write_path), omega, alph)}, PSNR = {PSNR(C, read_img(write_path))}")
    task_8(C, watermarks, alph)
    plot_imges(C)


if __name__ == '__main__':
    main()
