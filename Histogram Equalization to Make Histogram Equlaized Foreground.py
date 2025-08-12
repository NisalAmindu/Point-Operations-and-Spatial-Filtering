##Open and split the image into hue, saturation and value planes

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img_orig = cv.imread('a1images/jeniffer.jpg', cv.IMREAD_COLOR)
img_rgb = cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)

img_hsv = cv.cvtColor(img_orig, cv.COLOR_BGR2HSV)       # Convert the image into HSV color space
h_channel, s_channel, v_channel = cv.split(img_hsv)     # Split the converted image into hue, saturation and value planes

#region
plt.figure(figsize=(14, 6))

plt.subplot(141)
plt.imshow(img_rgb, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Displaying each separate plane in gray scale
plt.subplot(142)
plt.imshow(h_channel, cmap='gray')
plt.title('Hue Channel')
plt.axis('off')

plt.subplot(143)
plt.imshow(s_channel, cmap='gray')
plt.title('Saturation Channel')
plt.axis('off')

plt.subplot(144)
plt.imshow(v_channel, cmap='gray')
plt.title('Value Channel')
plt.axis('off')

plt.tight_layout()
plt.show()
#endregion


##Select the suitable plane to apply threshold to extract the foreground mask

# Select a threshold value randomly
threshold = 160

# Apply thresholding on three channels seperately
ret1, foreground_mask1 = cv.threshold(h_channel, threshold, 255, cv.THRESH_BINARY)
ret2, foreground_mask2 = cv.threshold(s_channel, threshold, 255, cv.THRESH_BINARY)
ret3, foreground_mask3 = cv.threshold(v_channel, threshold, 255, cv.THRESH_BINARY)

#region
plt.figure(figsize=(12,6))

plt.subplot(131)
plt.imshow(foreground_mask1, cmap='gray')
plt.title('Foreground Mask from Hue Plane')
plt.axis('off')

plt.subplot(132)
plt.imshow(foreground_mask2, cmap='gray')
plt.title('Foreground Mask from Saturation Plane')
plt.axis('off')

plt.subplot(133)
plt.imshow(foreground_mask3, cmap='gray')
plt.title('Foreground Mask from Value Plane')
plt.axis('off')

plt.show()
#endregion


##Obtain the forground only using cv.bitwise_and and compute the histogram

# Obtain the foreground using the mask from the value channel
foreground_img = cv.bitwise_and(img_orig, img_orig, mask=foreground_mask3)

#region
plt.figure(figsize=(14, 6))

plt.subplot(131)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(132)
plt.imshow(foreground_mask3, cmap='gray')
plt.title('Foreground Mask Obtained from the Value Plane')
plt.axis('off')

plt.subplot(133)
plt.imshow(cv.cvtColor(foreground_img, cv.COLOR_BGR2RGB))
plt.title('Foreground of the Original Image')
plt.axis('off')

plt.show()
#endregion

# Calculate histograms for foreground of each channel in BGR color space
b_hist = cv.calcHist([foreground_img], [0], None, [256], [0, 256])
g_hist = cv.calcHist([foreground_img], [1], None, [256], [0, 256])
r_hist = cv.calcHist([foreground_img], [2], None, [256], [0, 256])

#region
plt.figure(figsize=(14, 4))

plt.subplot(131)
plt.plot(b_hist, color='blue')
plt.title('Blue Channel')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(132)
plt.plot(g_hist, color='green')
plt.title('Green Channel')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(133)
plt.plot(r_hist, color='red')
plt.title('Red Channel')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.show()
#endregion


##Obtain the cumulative sum of the histogram

# Obtain the culmulative sum of the histogram
cumulative_hist_b = np.cumsum(b_hist)
cumulative_hist_g = np.cumsum(g_hist)
cumulative_hist_r = np.cumsum(r_hist)

#region
plt.figure(figsize=(14, 4))

plt.subplot(131)
plt.plot(cumulative_hist_b, color='blue')
plt.title('Cumulative Blue Channel')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')

plt.subplot(132)
plt.plot(cumulative_hist_g, color='green')
plt.title('Cumulative Green Channel')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')

plt.subplot(133)
plt.plot(cumulative_hist_r, color='red')
plt.title('Cumulative Red Channel')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')

plt.show()
#endregion


##Histogram-equalize the foreground by given formulas

# Histogram equalization for three color channels
r_equalized = cv.equalizeHist(foreground_img[:, :, 0])
g_equalized = cv.equalizeHist(foreground_img[:, :, 1])
b_equalized = cv.equalizeHist(foreground_img[:, :, 2])

# Merge the equalized channels
equalized_img = cv.merge((r_equalized, g_equalized, b_equalized))

# Calculate the histograms for channels
r_equalized_hist = cv.calcHist([equalized_img], [0], None, [256], [0, 256])
g_equalized_hist = cv.calcHist([equalized_img], [1], None, [256], [0, 256])
b_equalized_hist = cv.calcHist([equalized_img], [2], None, [256], [0, 256])

# Calculate the CDF for equalized channels
r_cumulative = np.cumsum(r_equalized_hist)
g_cumulative = np.cumsum(g_equalized_hist)
b_cumulative = np.cumsum(b_equalized_hist)

#region
plt.figure(figsize=(12, 8))

plt.subplot(121)
plt.imshow(cv.cvtColor(foreground_img, cv.COLOR_BGR2RGB))
plt.title('Foreground of the original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(cv.cvtColor(equalized_img, cv.COLOR_BGR2RGB))
plt.title('Foreground of the Equalized Image')
plt.axis('off')

plt.show()

plt.figure(figsize=(18, 4))

plt.subplot(131)
plt.plot(r_cumulative, color='red')
plt.title('Cumulative Histogram for Equalized Red Channel')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(132)
plt.plot(g_cumulative, color='green')
plt.title('Cumulative Histogram for Equalized Green Channel')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(133)
plt.plot(b_cumulative, color='blue')
plt.title('Cumulative Histogram for Equalized Blue Channel')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.show()
#endregion


##Add the original background with histogram equalized foreground

# Extract the background by bitwise_not
background = cv.bitwise_and(img_orig, img_orig, mask=cv.bitwise_not(foreground_mask3))

final_modified_img = cv.add(background, equalized_img)
final_modified_img_rgb = cv.cvtColor(final_modified_img, cv.COLOR_BGR2RGB)

#region
plt.figure(figsize=(16, 4))

plt.subplot(141)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(142)
plt.imshow(cv.cvtColor(background, cv.COLOR_BGR2RGB))
plt.title("Background of the Image")
plt.axis('off')

plt.subplot(143)
plt.imshow(cv.cvtColor(equalized_img, cv.COLOR_BGR2RGB))
plt.title('Foreground of the Equalized Image')
plt.axis('off')

plt.subplot(144)
plt.imshow(final_modified_img_rgb)
plt.title("Image after Foreground Equalization")
plt.axis('off')

plt.show()
#endregion