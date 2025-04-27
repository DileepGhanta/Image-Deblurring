import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt;


image = cv.imread('blur.png', cv.IMREAD_GRAYSCALE)


(x, y) = image.shape

K = (1 / 4.8976) * np.array([[0.3679, 0.6065, 0.3679], 
                             [0.6065, 1.0000, 0.6065], 
                             [0.3679, 0.6065, 0.3679]])

blurred = np.zeros_like(image, dtype=np.float32)

padim = np.pad(image, ((1, 1), (1, 1)), mode='edge')

for row in range(x):
    for col in range(y):
        subim = padim[row:row+3, col:col+3]
        blurred[row, col] = np.sum(subim *K)

blurred = np.clip(blurred, 0, 255).astype(np.uint8)


coin = image.astype(np.int32) - blurred.astype(np.int32)

coin_display = np.clip(coin, 0, 255).astype(np.uint8)

result = np.clip(image.astype(np.int32) + 5*(coin), 0, 255).astype(np.uint8)


fig, axs = plt.subplots(1, 3, figsize=(18, 6))

axs[0].imshow(image, cmap='gray')
axs[0].set_title('Original Image')

axs[1].imshow(blurred, cmap='gray')
axs[1].set_title('Original Image with Gaussian Blur')

axs[2].imshow(result, cmap='gray')
axs[2].set_title('Deblurred Image')


plt.show()
