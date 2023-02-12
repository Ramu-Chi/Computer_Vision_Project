import matplotlib.pyplot as plt
import numpy as np
import cv2
from math import ceil, floor

img_file_path = 'project1/image/sinus.png'

# -- Step 1: Pre-processing (convert to gray images) --
original_img = cv2.imread(img_file_path)
img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
H, W = len(img), len(img[0])

# -- Step 2: Noise reduction --
## Process sinusoidal noise 
# discrete fourier transform of the image
dft = np.fft.fft2(img)

# shift zero-frequency component to the center of the spectrum
dft_shift = np.fft.fftshift(dft)

# remove vertical sinusoidal noise
def remove_vertical_sinus(dft_shift, threshold=3):
    spectrum_mag = np.abs(dft_shift)
    crow, ccol = (H - 1) / 2, (W - 1) / 2
    x_axis_freqs = [np.mean([spectrum_mag[ceil(crow), i], spectrum_mag[floor(crow), i]]) for i in range(W)]

    mean = np.mean(x_axis_freqs)
    std = np.std(x_axis_freqs)
    z_scores = [(x - mean) / std for x in x_axis_freqs]

    for i in range(len(x_axis_freqs)):
        if i == ceil(ccol) or i == floor(ccol): continue
        if z_scores[i] > threshold:
            print('sinusoidal noise found')
            dft_shift[ceil(crow), i] = 0
            dft_shift[floor(crow), i] = 0
    
    return dft_shift

dft_shift = remove_vertical_sinus(dft_shift)

# inverse DFT
f_ishift = np.fft.ifftshift(dft_shift)
img = np.fft.ifft2(f_ishift)
img = np.real(img)

img = img.astype(np.uint8)

## Reduce salt and pepper noise: Median filter
img = cv2.medianBlur(img, 5)

## Erode to seperate objects
kernel = np.ones((3, 3), np.uint8)
img = cv2.erode(img, kernel)

## Process low images: Gamma correction
def gammaCorrection(src, gamma):
    table = [((i / 255) ** gamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)

img = gammaCorrection(img, 1/4)

## Linear stretching:
a_low, a_high = np.min(img), np.max(img)
for i in range(len(img)):
    for j in range(len(img[0])):
        img[i, j] = (img[i, j] - a_low) * 255 / (a_high - a_low)

## Reduce Gaussian noise
img = cv2.GaussianBlur(img, (7, 7), 1)
final = img

# -- Step 3: Counting objects --
## Detect edges: Canny algorithm
canny = cv2.Canny(final, 65, 200)

## Connect the edges: Dilation
kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], np.uint8)
dilated = cv2.dilate(canny, kernel)

## Calculate the contours:
contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# -- Step 4: Post-processing --
## Convert processed image to BGR (cv2 default format) to draw color contours
rgb = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

# Show each contour
# for i in contours:
#     contours_img = cv2.drawContours(rgb.copy(), [i], -1, (0, 0, 255), 2)
#     cv2.imshow("One contour", contours_img)
#     cv2.waitKey()

## Draw the contours for visualizing
contours_img = cv2.drawContours(rgb, contours, -1, (0, 0, 255), 1)

## Display the images:
# Stacking images side-by-side
res = np.hstack((original_img, contours_img))
print("Number of objects: ", len(contours))

cv2.imshow("Number of objects: " + str(len(contours)), res)
cv2.waitKey()
