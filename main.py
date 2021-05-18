from PCA_detect import PCA_facial_detection
from PCA_detect_utils import load_images_from_folder
import numpy as np
import matplotlib.pyplot as plt
import os

""""""
# Read in all of the images for the neutral expression
directory = os.getcwd()
image_folder_n = "/neutral"
images_neutral = load_images_from_folder(directory + image_folder_n)
# Read in all of the images for the smiling expression
image_folder_s = "/smiling"
images_smiling = load_images_from_folder(directory + image_folder_s)
# Go to lines 62 and 65 of PCA_detect.py to modify the folders for the rotation image and the nonhuman image
""""""

# Plot for the singular values of the data matrix
_, _, eig_a = PCA_facial_detection(images_neutral, select=-3, plot=False, debug=True)
plt.plot(range(0, 188), eig_a[0:188])
plt.show()


# Facial reconstruction for neutral image
rec_b, mse_b, _ = PCA_facial_detection(images_neutral, select=133, plot=False)
# Plot MSE vs k PCs
plt.plot(range(0, 188), mse_b)
plt.show()
# Show reconstruction
# print(rec_b[np.argmin(mse_b)].reshape(193,162).tolist())
plt.imshow(rec_b[np.argmin(mse_b)].reshape(
    193, 162), cmap='gray', vmin=0, vmax=1)
plt.show()


# Facial reconstruction for smiling image
rec_c, mse_c, _ = PCA_facial_detection(images_smiling, select=133, plot=False)
# Plot MSE vs k PCs
plt.plot(range(0, 188), mse_c)
plt.show()
# Show reconstruction
plt.imshow(rec_c[np.argmin(mse_c)].reshape(
    193, 162), cmap='gray', vmin=0, vmax=1)
plt.show()


# Facial reconstruction for unused neutral image
rec_d, mse_d, _ = PCA_facial_detection(images_neutral, select=193, plot=False)
# Plot MSE vs k PCs
plt.plot(range(0, 188), mse_d)
plt.show()
# Show reconstruction
plt.imshow(rec_d[np.argmin(mse_d)].reshape(
    193, 162), cmap='gray', vmin=0, vmax=1)
plt.show()


# Facial reconstruction on non-human image
rec_e, _, _ = PCA_facial_detection(images_neutral, select=-1, plot=False)
# Show reconstruction
plt.imshow(rec_e[0].reshape(
    193, 162), cmap='gray', vmin=0, vmax=1)
plt.show()


# Facial reconstruction on rotated neutral image
rec_f, _, _ = PCA_facial_detection(images_neutral, select=-2, plot=False)
# Show reconstruction for some rotations
plt.imshow(rec_f[0].reshape(
    193, 162), cmap='gray', vmin=0, vmax=1)
plt.show()
plt.imshow(rec_f[4].reshape(
    193, 162), cmap='gray', vmin=0, vmax=1)
plt.show()
plt.imshow(rec_f[9].reshape(
    193, 162), cmap='gray', vmin=0, vmax=1)
plt.show()
plt.imshow(rec_f[18].reshape(
    193, 162), cmap='gray', vmin=0, vmax=1)
plt.show()
plt.imshow(rec_f[27].reshape(
    193, 162), cmap='gray', vmin=0, vmax=1)
plt.show()
