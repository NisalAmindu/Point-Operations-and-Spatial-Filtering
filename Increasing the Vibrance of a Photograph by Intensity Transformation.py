import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from ipywidgets import interactive

def vibrance(x, a, sigma=70):
    return int(min(x + (a*128)*np.exp((-(x-128)**2)/(2*(sigma**2))), 255))  # Transformation function

def transform(a):
    # This function will apply the desired transformation to selected planes of the image
    plt.clf()
    table = np.array([vibrance(x, a) for x in np.arange(0, 256)]).astype('uint8')
    s_channel_corrected = cv.LUT(s_channel, table)          # Apply vibrance correction to the saturation plane
    img_corrected = cv.merge((h_channel, s_channel_corrected, v_channel))   # Merge corrected plane with hue and value planes
    img_corrected_rgb = cv.cvtColor(img_corrected, cv.COLOR_HSV2RGB)
    #region
    plt.figure(figsize=(12,12))
    plt.subplot(121)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(img_corrected_rgb)
    plt.title('Intensity Transformed Image')
    plt.axis('off')
    
    plt.show()
    
    a_value = np.arange(0, 256)
    plt.figure(figsize=(4,4))
    plt.plot(a_value,table, label=f'a = {a}')
    plt.title('Intensity Transformation Function')
    plt.xlabel('Input Intensity')
    plt.ylabel('Transformed Intensity')
    plt.legend()
    plt.grid()
    
    plt.show()
    #endregion

img_orig = cv.imread('a1images/spider.png', cv.IMREAD_COLOR)
img_rgb = cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)
img_hsv = cv.cvtColor(img_orig, cv.COLOR_BGR2HSV)           # Convert the image into HSV color space
h_channel, s_channel, v_channel = cv.split(img_hsv)         # Split the image into hue, saturation and value planes

# Interactive Slider
final_plot = interactive(transform, a=(0, 1, 0.001))
output = final_plot.children[-1]
output.layout.height = '800px'
final_plot