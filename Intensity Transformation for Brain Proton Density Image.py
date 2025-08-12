import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img_orig = cv.imread('a1images/brain_proton_density_slice.png', cv.IMREAD_GRAYSCALE)

white_threshold_upper = 255
white_threshold_lower = 175
white_matter = np.linspace(0, 255, 256, dtype='uint8')
white_matter[:white_threshold_lower] = 0
white_matter[white_threshold_lower:] = np.linspace(white_threshold_lower, white_threshold_upper, white_threshold_upper - white_threshold_lower + 1, dtype='uint8')

gray_threshold_upper = 180
gray_threshold_lower = 138
gray_matter = np.linspace(0, 255, 256, dtype='uint8')
gray_matter[:gray_threshold_lower] = 0
gray_matter[gray_threshold_upper:] = 0
gray_matter[gray_threshold_lower:gray_threshold_upper] = np.linspace(gray_threshold_lower, gray_threshold_upper, gray_threshold_upper - gray_threshold_lower, dtype='uint8')

white_transform = cv.LUT(img_orig, white_matter)
gray_transform = cv.LUT(img_orig, gray_matter)


fig, ax = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(12,4))

ax[0].plot(white_matter)
ax[0].set_xlabel(r'Input, $f(\mathbf{x})$')
ax[0].set_ylabel(r'Output, $\mathrm{T}[f(\mathbf{x})]$')
ax[0].set_xlim(0, 255)
ax[0].set_ylim(0, 255)
ax[0].set_aspect('equal')
ax[0].set_title('Intensity Transformation: White Matter')


ax[1].plot(gray_matter)
ax[1].set_xlabel(r'Input, $f(\mathbf{x})$')
ax[1].set_ylabel(r'Output, $\mathrm{T}[f(\mathbf{x})]$')
ax[1].set_xlim(0, 255)
ax[1].set_ylim(0, 255)
ax[1].set_aspect('equal')
ax[1].set_title('Intensity Transformation: Gray Matter')

plt.show()

fig, ax = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(12,6))

ax[0].imshow(img_orig, cmap="gray")
ax[0].set_title('Original Image')

ax[1].imshow(white_transform, cmap="gray")
ax[1].set_title('Intensity Transformation: White Matter')

ax[2].imshow(gray_transform, cmap="gray")
ax[2].set_title('Intensity Transformation: Gray Matter')