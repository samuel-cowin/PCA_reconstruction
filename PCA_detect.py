from PCA_detect_utils import reconstruction, compute_mse, image_resize, rotate_image, load_images_from_folder
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def PCA_facial_detection(images, select=0, plot=False, debug=False):
    # Compute all m image vectors with each image vector being N^2 x 1
    dimy = len(images[0])
    dimx = len(images[0][0])
    images_vector = np.transpose(np.array([im.reshape(dimx*dimy) for im in images]))

    # Compute the average face vector and subtract from each vector
    phi = (1/len(images_vector[0:189]))*np.sum(images_vector[:, 0:189], 1)
    normed_images_vector = np.transpose(np.transpose(images_vector) - phi)

    # Compute the m x m covariance matrix to minimize the computational burden
    cov_matrix = (1/len(images_vector[0:189]))*np.dot(np.transpose(normed_images_vector[:, 0:189]),
                        normed_images_vector[:, 0:189])

    # Compute the sorted eigenvalues for the lower dimensional covariance matrix
    # Simultaneously compute the corresponding eigenvectors
    # w[i] eigenvalue corresponds to the v[:,i] eigenvector
    eig_values_m, eig_vectors_m = np.linalg.eig(cov_matrix)

    # Map the lower dimensional eigenvalues into the higher dimensional space
    # These eigenvalues correspond to the m largest eigenvalues of the higher space
    eig_vectors_nsq = np.dot(normed_images_vector[:, 0:189], eig_vectors_m)

    # Normalize the eigenvectors for the new space
    norm = np.sqrt(np.sum(eig_vectors_nsq**2, 0))
    normed_eig_vectors_nsq = eig_vectors_nsq/norm

    # Print statements for understanding dimensionality
    # Dimx: 162
    # Dimy: 193
    # Image vectors: (31266, 200)
    # Phi: (31266,)
    # Shape of normed image vectors: (31266, 200)
    # Cov Matrix: (189, 189)
    # Eigen values in m: (189,)
    # Eigen Vectors in m: (189,)
    # Eigen vectors in nsq: (31266, 189)
    # Normalized eigenvectors in nsq: (189,)
    # Reconstructed Images: (188, 31266)
    if debug:
        print("Dimx: {}".format(dimx))
        print("Dimy: {}".format(dimy))
        print("Image vectors: {}".format(np.shape(images_vector)))
        print("Phi: {}".format(np.shape(phi)))
        print("Shape of normed image vectors: {}".format(np.shape(normed_images_vector)))
        print("Cov Matrix: {}".format(np.shape(cov_matrix)))
        print("Eigen values in m: {}".format(np.shape(eig_values_m)))
        print("Eigen Vectors in m: {}".format(np.shape(eig_vectors_m[0])))
        print("Eigen vectors in nsq: {}".format(np.shape(eig_vectors_nsq)))
        print("Normalized eigenvectors in nsq: {}".format(np.shape(np.sum(normed_eig_vectors_nsq**2,0))))

    # Select the image that will be reconstructed
    if select == -1:
        directory = os.getcwd()
        img = load_images_from_folder(directory+"/nonhuman")
    elif select == -2:
        directory = os.getcwd()
        img = load_images_from_folder(directory+"/rotate")
    else:
        if select!= -3: 
            images_vector_plotting = (images_vector[:, select] * 255).astype(np.uint8)
            plt.imshow(images_vector_plotting.reshape(
                193, 162), cmap='gray', vmin=0, vmax=255)
            plt.show()

    # Use the k largest eigenvalues to perform reconstruction and compute MSE
    reconstructed_ims = []
    mse = []
    if select == -1:
        k = 189
        image_input = image_resize(img[0], 0, 1)
        plot_car = (image_input * 255).astype(np.uint8)
        plot_car = cv2.resize(np.float32(plot_car), dsize=(162, 193))
        plt.imshow(plot_car, cmap='gray', vmin=0, vmax=255)
        plt.show()
        reconstructed_ims.append(reconstruction(
            normed_eig_vectors_nsq[:, 0:189], image_input.reshape(dimx*dimy), phi, k=k, plot=plot))
    elif select == -2:
        k = 189
        plots = [0, 4, 9, 18, 27]
        to_plot = False
        for i in range(36):
            if i in plots:
                to_plot = True
            rotated_im = rotate_image(img[0], i*10, plot=to_plot)
            rotated_im_input = image_resize(rotated_im, 0, 1)
            reconstructed_ims.append(reconstruction(
                normed_eig_vectors_nsq[:, 0:189], rotated_im_input.reshape(dimx*dimy), phi, k=k, plot=plot))
            to_plot = False
    else:
        for k in range(1, 189):
            reconstructed_ims.append(reconstruction(
                normed_eig_vectors_nsq[:, 0:189], images_vector[:, select], phi, k=k-1, plot=plot))
            mse.append(compute_mse(
                reconstructed_ims[k-1], images_vector[:, select])/(dimx*dimy))
        if debug:
            print("Reconstructed Images: {}".format(np.shape(reconstructed_ims)))
    return reconstructed_ims, mse, sorted(np.sqrt(eig_values_m), reverse=True)
