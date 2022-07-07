import numpy as np
import cv2
from matplotlib import pyplot as plt

image = cv2.imread('linearnew2.jpg', 0) #можно линеар 1
float_image = np.float32(image)

fft_image = np.fft.fft2(float_image)
fft_image = np.fft.fftshift(fft_image)
print('Data shape and type:   ', fft_image.shape, fft_image.dtype)
fft_image_processed = fft_image.copy() 

fft_image_processed[0:235, 196:201] = 0 
fft_image_processed[240:477, 196:201] = 0
fft_image_processed[0:235, 275:280] = 0
fft_image_processed[240:478, 275:280] = 0

fft_image_processed[210:215, :] = 0 
fft_image_processed[265:270, :] = 0
fft_image_processed[0:235, 280:478] = 0
fft_image_processed[0:235, 0:201] = 0
fft_image_processed[240:477, 280:478] = 0
fft_image_processed[240:477, 0:201] = 0



'''fft_image_processed[:, 196:201] = 0 
fft_image_processed[:, 275:280] = 0
fft_image_processed[210:215,:] = 0
fft_image_processed[265:270,:] = 0'''


#magnitude = np.log(np.abs(fft_image))
#magnitude = np.log(np.abs(fft_image))

magnitude = np.abs(fft_image_processed)
s_min = magnitude.min()
s_max = magnitude.max()

print(s_min, s_max)
#bin_counts, bin_edges, patches = plt.hist(magnitude.ravel())

reconstructed_image_complex = np.fft.ifft2(fft_image_processed)
reconstructed_image = np.abs(reconstructed_image_complex)

plt.figure(figsize=(10,10))
plt.imshow(np.log(magnitude))
#plt.imshow(reconstructed_image, cmap = 'gray')
plt.show()