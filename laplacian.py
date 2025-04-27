import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

k = np.array([[1, 1, 1], 
              [1, -8, 1], 
              [1, 1, 1]], dtype=np.int32) 

image = cv.imread("blur.png", cv.IMREAD_GRAYSCALE)

(x, y) = image.shape
laplacian = np.zeros_like(image, dtype=np.int32)

padim = np.pad(image, ((1, 1), (1, 1)), mode='edge')


for row in range(x):
    for col in range(y):
        subim = padim[row:row+3, col:col+3]
        laplacian[row, col] = np.sum(subim * k)

laplacian = np.clip(laplacian, -255, 255)


alpha = 0.3 
deblurred = image - alpha * laplacian
print(" Data type of deblurred" ,deblurred.dtype)
deblurred = np.clip(deblurred, 0, 255).astype(np.uint8)


fig, axs = plt.subplots(1, 3, figsize=(18, 6))

axs[0].imshow(image, cmap='gray')
axs[0].set_title('Original Image')

axs[1].imshow(np.clip(laplacian, 0, 255).astype(np.uint8), cmap='gray')
axs[1].set_title('Laplacian (Edges)')

axs[2].imshow(deblurred, cmap='gray')
axs[2].set_title('Deblurred Image')


plt.show()
