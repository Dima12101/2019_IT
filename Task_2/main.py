import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def convolution(image, kernel, div=1):
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise Exception("Kernel don't must have the even size")

    vSh_im = image.shape[0]
    hSh_im = image.shape[1]
    newImageXY = np.zeros((vSh_im, hSh_im, 3))
    newImageX = np.zeros((vSh_im, hSh_im, 3))
    newImageY = np.zeros((vSh_im, hSh_im, 3))

    vSh_k = kernel.shape[0]
    hSh_k = kernel.shape[1]
    for i in range(vSh_im):
        for j in range(hSh_im):
            # Sizes of area
            up = min(vSh_k // 2, i)
            down = min(vSh_k // 2, vSh_im - i - 1)
            left = min(hSh_k // 2, j)
            right = min(hSh_k // 2, hSh_im - j - 1)

            # Using slicing
            imageArea = image[i - up:i + down + 1, j - left:j + right + 1]
            imageArea_r = imageArea[:, :, 0]
            imageArea_g = imageArea[:, :, 1]
            imageArea_b = imageArea[:, :, 2]

            kernelArea = kernel[(vSh_k // 2) - up:(vSh_k // 2) + down + 1,
                                (hSh_k // 2) - left:(hSh_k // 2) + right + 1]

            r_xy = sum(sum(imageArea_r * kernelArea))
            g_xy = sum(sum(imageArea_g * kernelArea))
            b_xy = sum(sum(imageArea_b * kernelArea))

            r_x = sum((imageArea_r * kernelArea)[vSh_k // 2, :])
            g_x = sum((imageArea_g * kernelArea)[vSh_k // 2, :])
            b_x = sum((imageArea_b * kernelArea)[vSh_k // 2, :])

            r_y = sum((imageArea_r * kernelArea)[:, hSh_k // 2])
            g_y = sum((imageArea_g * kernelArea)[:, hSh_k // 2])
            b_y = sum((imageArea_b * kernelArea)[:, hSh_k // 2])

            # Using cycle
            # r, g, b = 0, 0, 0
            # i_k = (vSh_k // 2) - up
            # for i_im in range(i - up, i + down + 1):
            #     j_k = (hSh_k // 2) - left
            #     for j_im in range(j - left, j + right + 1):
            #         r += image[i_im][j_im][0] * kernel[i_k][j_k]
            #         g += image[i_im][j_im][1] * kernel[i_k][j_k]
            #         b += image[i_im][j_im][2] * kernel[i_k][j_k]
            #         j_k += 1
            #     i_k += 1

            newImageXY[i][j] = [abs(r_xy) / div, abs(g_xy) / div, abs(b_xy) / div]
            newImageX[i][j] = [abs(r_x) / div, abs(g_x) / div, abs(b_x) / div]
            newImageY[i][j] = [abs(r_y) / div, abs(g_y) / div, abs(b_y) / div]
    return newImageXY, newImageX, newImageY


def gauseKernel(sigma, h, w):
    kernel = np.zeros((h, w))
    n, m = h // 2, w // 2
    i = 0
    for l in range(-n, n + 1):
        j = 0
        for k in range(-m, m + 1):
            d = np.sqrt(l ** 2 + k ** 2)
            kernel[i][j] = np.exp(-((d ** 2) / (2 * (sigma ** 2)))) / (np.sqrt(2 * np.pi) * sigma)
            j += 1
        i += 1
    return kernel

img = mpimg.imread('cat.png')

# kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
# kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) #up contrast
# kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) #box filter

#kernel = gauseKernel(sigma=4, h=7, w=7)
#newImageXY, newImageX, newImageY = convolution(img, kernel)

#plt.imshow(newImageXY)
#plt.imshow(newImageX)
#plt.imshow(newImageY)
#plt.show()


