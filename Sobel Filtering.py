##Using the Existing Filter2D Function

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

img = cv.imread('a1images/einstein.png', cv.IMREAD_GRAYSCALE)

kernel = np.array([(1, 0, -1), (2, 0, -2), (1, 0, -1)], dtype='float')
img_filt = cv.filter2D(img, -1, kernel)

fig, ax = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(12,12))

ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original Image')
ax[0].set_xticks([]), ax[0].set_yticks([])

ax[1].imshow(img_filt, cmap='gray')
ax[1].set_title('Sobel Filtered Image')
ax[1].set_xticks([]), ax[0].set_yticks([])

plt.show()


## Using a Custom Code to Sobel Filter the Image

def sobel_filter(image, kernel):
    # Sobel filtering image using convolution
    img_height, img_width = image.shape
    
    kernel_size = kernel.shape[0]
    
    output_img = np.zeros((img_height - kernel_size + 1, img_width - kernel_size + 1))
    
    for i in range(output_img.shape[0]):
        for j in range(output_img.shape[1]):
            region = image[i : i + kernel_size, j : j + kernel_size]
            conv_result = np.sum(region * kernel)
            output_img[i, j] = conv_result
            #region
            # Apply pixel value matching 
            if output_img[i, j] < 0:       
                output_img[i, j] = 0
            elif output_img[i, j] > 255:   
                output_img[i, j] = 255
            #endregion
    return output_img.astype(np.uint8)

img_orig = cv.imread('a1images/einstein.png', cv.IMREAD_GRAYSCALE)
# Introducing the kernels which will be convoluted with the image.
x_kernel = np.array([(-1, 0, 1), (-2, 0, 2), (-1, 0, 1)], dtype='float')
y_kernel = np.array([(-1, -2, -1), (0, 0, 0), (1, 2, 1)], dtype='float')

# Introduce sobel kernels for horizontal and vertical directions
sobel_x = sobel_filter(img_orig, x_kernel)
sobel_y = sobel_filter(img_orig, y_kernel)

# Calculate the magnitude of the gradient
gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)

#region
plt.figure(figsize=(12,6))
plt.suptitle('Filtering Using Custom Sobel Filter')

plt.subplot(121)
plt.imshow(cv.cvtColor(img_orig, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(gradient_mag, cmap='gray')
plt.title('Sobel Filtered Image')
plt.axis('off')

plt.show()
#endregion


##Using the Associative Property of Convolution

img = cv.imread('a1images/einstein.png', cv.IMREAD_GRAYSCALE)

column_kernel = np.array([(1), (2), (1)], dtype='float')   
row_kernel = np.array([(1, 0, -1)], dtype='float')

conv_1 = cv.filter2D(img, -1, column_kernel)    # First Convolute with the column vector
conv_2 = cv.filter2D(conv_1, -1, row_kernel)    # Convolute the above result with the row vector

#region
fig, axes  = plt.subplots(1,2, sharex='all', sharey='all', figsize=(12,12))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].set_xticks([]), axes[0].set_yticks([])

axes[1].imshow(conv_2, cmap='gray')
axes[1].set_title('Custom Sobel Filtered Image')
axes[1].set_xticks([]), axes[1].set_yticks([])

plt.show()
#endregion


##Final Sobel Filtered Images


plt.figure(figsize=(16,4))

plt.subplot(141)
plt.imshow(img, cmap='gray')
plt.title('Original Image in Gray Scale', fontsize=8)
plt.axis('off')

plt.subplot(142)
plt.imshow(img_filt, cmap='gray')
plt.title('Sobel Filtered Using filter2D Function', fontsize=8)
plt.axis('off')

plt.subplot(143)
plt.imshow(gradient_mag, cmap='gray')
plt.title('Sobel Filtered Using Custom Convolution Method', fontsize=8)
plt.axis('off')

plt.subplot(144)
plt.imshow(conv_2, cmap='gray')
plt.title('Sobel Filtered Using Assosiative Convolution', fontsize=8)
plt.axis('off')

plt.show()