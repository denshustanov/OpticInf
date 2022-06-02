import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import cv2
import os
from skimage.metrics import structural_similarity as ssim
matplotlib.use('TkAgg')


def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)


def calculate_i2fft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.ifft2(ft)
    return np.fft.fftshift(ft)

# #
# im = plt.imread('cat.jpg')
# im = im[:, :, :3].mean(axis=2)
# fig, ax = plt.subplots(1, 2)
# plt.set_cmap('gray')
# res = calculate_2dft(im)
# # ax[0].imshow(im)
# # ax[1].imshow(np.log(np.abs(res)))
# # plt.show()
# a = np.zeros(res.shape, dtype='complex')
# cx = res.shape[0]//2
# cy = res.shape[1]//2
#
# x = []
# s = []
#
# for i in range(0, cx):
#     print(i)
#     x.append(i)
#     a[cx-i:cx+i, cy-i:cy+i] += res[cx-i:cx+i, cy-i:cy+i]
#     # ifft = np.real(calculate_i2fft(a))
#     # s.append(ssim(ifft, im))
#     ax[0].imshow(np.log(np.abs(a)))
#     ax[1].imshow(np.real(calculate_i2fft(a)))
#     a = np.zeros(res.shape, dtype='complex')
#     plt.savefig('cat/'+str(i)+'.png')
# #
# plt.plot(x, s)
# plt.show()

files = os.listdir('cat')
files = sorted(files, key=lambda x: int(x[:x.find('.')]), reverse=False)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter('cat.mp4', fourcc, 5, (640, 480))
for file in files:
    # print(file)
    im = cv2.imread('cat/'+file)
    print(file, im.shape)
    # cv2.imshow(file, im)
    writer.write(im)
writer.release()

