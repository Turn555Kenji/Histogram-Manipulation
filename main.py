import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("dog2.png", cv2.IMREAD_GRAYSCALE)

mode = "equalization"  # "normalization", "negative", "binarization"
threshold = 128

if mode == "equalization":
    K1, K2, L1, L2 = 0, 1, 0, 255
elif mode == "normalization":
    K1, K2 = 0, 1
    L1, L2 = int(img.min()), int(img.max())
elif mode == "negative":
    K1, K2 = 1, 0
    L1, L2 = int(img.min()), int(img.max())
elif mode == "binarization":
    K1, K2 = 0, 1
    L1, L2 = threshold, threshold


# H1
H1 = np.zeros(256, dtype=int)

for pixel in img.flatten():
    H1[pixel] += 1

# H2
H2 = np.zeros(256, dtype=int)
H2[0] = H1[0]

for i in range(1, 256):
    H2[i] = H2[i - 1] + H1[i]

# Normalize CDF (H2)
H2_norm = H2 / H2[255]

# H3
x = np.arange(256)
if L1 != L2:
    a = (K2 - K1) / (L2 - L1)
else:
    a = 0
b = K1
H3 = a*x + b

# Value assignment/lookup
map_table = np.zeros(256, dtype=np.uint8)

for l in range(256):
    if mode == "equalization": #Maps H2_norm value according to H3 (255)
        map_table[l] = round(H2_norm[l] * 255)
    else: #Linear Transformations
        if l < L1:
            k = K1
        elif l >= L2:
            k = K2
        else:
            k = ((K2 - K1) / (L2 - L1)) * (l - L1) + K1
        map_table[l] = round(k * 255)

# H5
map = map_table[img]
H5 = np.zeros(256, dtype=int)

for pixel in map.flatten():
    H5[pixel] += 1

# Plotting
fig, ax = plt.subplots(3, 2)

ax[0][0].imshow(img, cmap='gray', vmin=0, vmax=255)
#ax[1][0].hist(img.flatten(), bins=256, range=[0,256], alpha=0.3, label='Histogram')
ax[1][0].bar(x, H1, width=3.0)
ax[2][0].bar(x, H2_norm, width=3.0)

ax[0][1].imshow(map, cmap='gray', vmin=0, vmax=255)
ax[1][1].plot(x, H3, label='Ideal')
#ax[1][1].hist(map.flatten(), bins=256, range=[0,256], alpha=0.3, label='Histogram')
ax[2][1].bar(x, H5, width=3.0)

plt.show()