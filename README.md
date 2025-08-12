# EN3160 Assignment 1 – Intensity Transformations & Neighborhood Filtering

This repository contains my solutions for **EN3160 Assignment 1**.  
The tasks involve various image processing techniques including intensity transformations, histogram equalization, color space operations, Sobel filtering, and image zooming.

## Implemented Tasks
1. **Intensity Transformation** – Applied given mapping to sample images.
2. **Brain Image Processing** – Enhanced white and gray matter using intensity transformations.
3. **Gamma Correction** – Applied to L channel in L\*a\*b\* color space and compared histograms.
4. **Vibrance Enhancement** – Adjusted saturation using provided formula and parameter tuning.
5. **Histogram Equalization** – Manual implementation without built-in functions.
6. **Foreground Histogram Equalization** – Applied only to masked foreground region.
7. **Sobel Filtering** – Implemented using:
   - `cv2.filter2D`
   - Custom convolution
   - Separable convolution
8. **Image Zooming** – Nearest-neighbor and bilinear interpolation, SSD comparison.
9. **Foreground Extraction & Background Blur** – Used grabCut and Gaussian blur.

## Requirements
- Python 3.8+
- Install dependencies:
```bash
pip install numpy opencv-python matplotlib 
