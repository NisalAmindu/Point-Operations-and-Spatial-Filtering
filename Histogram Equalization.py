import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

img = cv.imread('a1images/shells.tif', cv.IMREAD_GRAYSCALE)

hist,  bins = np.histogram(img.ravel(), 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/cdf.max()
plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf, histogram'), loc='upper left')
plt.title('Histogram of the Original Image')
plt.show()

equ = cv.equalizeHist(img)
hist, bins = np.histogram(equ.ravel(), 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf*hist.max()/cdf.max()

plt.plot(cdf_normalized, color='b')
plt.hist(equ.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.title('Histogram of the Equalized Image')
plt.show()

res = np.hstack((img, equ))
plt.axis('off')
plt.imshow(res, cmap='gray')

#Equalization using custom equalization function


def equalize(image):
    histogram,  bins = np.histogram(img.flatten(), bins=256, range=(0, 256))    # Calculate the histogram of the image
    cdf = hist.cumsum()                                                         # calculate cumulative dixtribution function
    cdf_normalized = cdf * histogram.max()/cdf.max()                            # Normalize the CDF to map the range into (0-255) range
    table = np.interp(image.flatten(), bins[:-1], cdf_normalized)               # Store the mapped values into a table                 
    equalized_image = table.reshape(image.shape)                                # Reshape the table into the shape of the original image
    return equalized_image.astype('uint8')
    

hist,  bins = np.histogram(img.ravel(), 256, [0, 256])      # Calculate the histogram of the original image
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/cdf.max()
#region
plt.figure(figsize=(12,4))

plt.subplot(121)
plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf, histogram'), loc='upper left')
plt.title('Histogram of the Original Image')
#endregion

equ = equalize(img)
hist, bins = np.histogram(equ.ravel(), 256, [0, 256])       # Calculate the histogram of the equalized image
cdf = hist.cumsum()
cdf_normalized = cdf*hist.max()/cdf.max()
#region
plt.subplot(122)
plt.plot(cdf_normalized, color='b')
plt.hist(equ.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.title('Histogram of the Equalized Image')
plt.show()
#endregion

conversion = np.hstack((img, equ))
plt.axis('off')
plt.imshow(conversion, cmap='gray')