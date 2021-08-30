from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt 
from time import time
from skimage import io
from filters import *

img = io.imread('dog.jpg', as_gray = True)
# 5x5 Gaussian blur
kernel = np.array(
    [
        [1,4,6,4,1],
        [4,16,24,16,4],
        [6,24,36,24,6],
        [4,16,24,16,4],
        [1,4,6,4,1]
    ]
)
t0 = time()
out = conv_nested(img, kernel)
t1=time()
t_normal = t1-t0



#plot original image
plt.subplot(1,2,1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')


#plot convolved image
plt.subplot(1,2,2)
plt.imshow(out)
plt.title('Blurred')
plt.axis('off')

plt.show()