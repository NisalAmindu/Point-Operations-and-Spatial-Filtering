##Zooming by Nearest-Neighbour Method

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img_orig = cv.imread('a1images/im03small.png', cv.IMREAD_COLOR)
img_rgb = cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)

def ssd(img1, img2):
    # Calculate the sum of squared difference
    return np.sum((img1 - img2)**2)

def zooming(original_image, zoom_factor):
    height, width, channels = original_image.shape
    
    # Apply zooming factor and prepare the shape of the zoomed image
    zoomed_height = int(height*zoom_factor) - 1
    zoomed_width = int(width*zoom_factor)
    zoomed_image = np.zeros((zoomed_height, zoomed_width, channels), dtype=np.uint8)
    
    for i in range(zoomed_height):
        for j in range(zoomed_width):
            # Zooming operation pixelwise implementation
            zoomed_image[i, j] = original_image[int(i/zoom_factor), int(j/zoom_factor)]
            
    print("Shape of original image: ", original_image.shape)
    print("Shape of the zoomed image: ", zoomed_image.shape)
    
    zoomed_image_rgb = cv.cvtColor(zoomed_image, cv.COLOR_BGR2RGB)
    
    #region
    plt.figure(figsize=(10,6))
    
    plt.subplot(121)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(zoomed_image_rgb)
    plt.title('Zoomed Image')
    plt.axis('off')
    
    plt.show()
    #endregion
    
    cv.imwrite('im03small.png' + 'nearest_neighbour_zooming.png', zoomed_image)         # Write zoomed image into a new file
    
    img_BIG = cv.imread('a1images/im03.png', cv.IMREAD_COLOR)
    print("SSD value between original and zoomed images: ", ssd(img_BIG, zoomed_image)) # Calculate the SSD value between zoomed image and the original image
    
    
    
zoom_factor = 4
print("Zooming factor: ", zoom_factor)
zooming(img_orig, zoom_factor)


##Zooming by Bilinear Interpolation

img_orig = cv.imread('a1images/im02small.png', cv.IMREAD_COLOR)
img_rgb = cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)

def ssd(img1, img2):
    return np.sum((img1 - img2)**2)

def zooming(original_image, zoom_factor):
    # Zooming by bilinear interpolation method
    height, width, channels = original_image.shape
    zoomed_height = int(height*zoom_factor)
    zoomed_width = int(width*zoom_factor)
    zoomed_image = np.zeros((zoomed_height, zoomed_width, channels), dtype=np.uint8)
    
    y_scale = height / zoomed_height
    x_scale = width / zoomed_width
    
    for i in range(zoomed_height):
        for j in range(zoomed_width):
            original_y = i * y_scale
            original_x = j * x_scale
            
            # Calculate the four nearest neighbours
            x1, y1 = int(original_x), int(original_y)
            x2, y2 = x1 + 1, y1 + 1
            
            # check boundaries
            if x2 >= width: x2 = width - 1
            if y2 >= height: y2 = height - 1
            
            # Interpolation weights
            weight_x = original_x - x1
            weight_y = original_y - y1
            
            # Apply bilinear interpolation
            pixel_interpolated = ((1 - weight_x) * (1 - weight_y) * original_image[y1, x1] +  weight_x * (1 - weight_y) * original_image[y1, x2] + 
                                  (1 - weight_x) * weight_y * original_image[y2, x1] +  weight_x * weight_y * original_image[y2, x2])
            
            # Set the pixel value in zoomed image
            zoomed_image[i, j] = pixel_interpolated
            
    print("Shape of original image: ", original_image.shape)
    print("Shape of the zoomed image: ", zoomed_image.shape)
    
    zoomed_image_rgb = cv.cvtColor(zoomed_image, cv.COLOR_BGR2RGB)
    
    #region
    plt.figure(figsize=(10,6))
    
    plt.subplot(121)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(zoomed_image_rgb)
    plt.title('Zoomed Image')
    plt.axis('off')
    
    plt.show()
    #endregion
    
    
    img_BIG = cv.imread('a1images/im02.png', cv.IMREAD_COLOR)
    print("SSD value between original and zoomed images: ", ssd(img_BIG, zoomed_image))
    
    return zoomed_image

zoom_factor = 4
print('Zooming Factor: ', zoom_factor)
zoomed_image = zooming(img_orig, zoom_factor)
cv.imwrite('im02small.png' + 'zoomed_by_bilinear_interpolation.png', zoomed_image)