import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

# Load image and convert to grayscale float32 for better precision
img = cv2.imread('cameraman.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)

# Add Gaussian noise
mean = 0
stddev = 10  # reduce stddev for less severe noise
noise = np.random.normal(mean, stddev, img.shape)
noisy_img = img + noise
noisy_img = np.clip(noisy_img, 0, 255)

# Apply Wiener filter
filtered_img = wiener(noisy_img, (5, 5))  # small kernel works better for mild noise

# Convert all images to uint8 for display
original_uint8 = img.astype(np.uint8)
noisy_uint8 = noisy_img.astype(np.uint8)
filtered_uint8 = np.clip(filtered_img, 0, 255).astype(np.uint8)

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(original_uint8, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Noisy Image")
plt.imshow(noisy_uint8, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Wiener Filtered")
plt.imshow(filtered_uint8, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

