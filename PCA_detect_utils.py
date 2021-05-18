import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import re
import os


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder), key=natural_keys):
        img = cv2.imread(os.path.join(folder, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray is not None:
            gray = gray.astype(np.float32)
            gray /= 255.
            images.append(gray)
    return images


def reconstruction(eig_vectors, im_vector, mean, k=10, plot=False):
    gamma_hat = 0
    for col_vector in range(k):
        gamma_hat += np.dot(compute_coefficient(np.transpose(
            eig_vectors[:, col_vector]), im_vector-mean), eig_vectors[:, col_vector])
    if plot:
        reconstructed_im = gamma_hat + mean
        plt.imshow(reconstructed_im.reshape(193, 162),
                   cmap='gray', vmin=0, vmax=255)
        plt.show()
    return gamma_hat + mean


def compute_coefficient(eig_vector_T, gamma):
    return np.dot(eig_vector_T, gamma)


def compute_mse(y, y_hat):
    difference = y-y_hat
    return np.sqrt(np.dot(difference, difference))


def image_resize(im, vector_bool, image_bool):
    if vector_bool:
        return cv2.resize(np.float32(im), dsize=(31266, 1))
    if image_bool:
        return cv2.resize(np.float32(im), dsize=(162, 193))


def rotate_image(im, degree, plot=False):
    rot_im = imutils.rotate_bound(im, angle=degree)
    if plot:
        plot_im = (rot_im * 255).astype(np.uint8)
        plot_rot = cv2.resize(np.float32(plot_im), dsize=(162, 193))
        plt.imshow(plot_rot, cmap='gray', vmin=0, vmax=255)
        plt.show()
    return rot_im
