import cv2

import matplotlib.pyplot as plt

image = cv2.imread('cameraman.jpg')
deblurred = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

fig, axs = plt.subplots(1, 2, figsize=(18, 6))

axs[0].imshow(image, cmap='gray')
axs[0].set_title('Original Image')

axs[1].imshow(deblurred, cmap='gray')
axs[1].set_title('Deblurred image')

plt.show()