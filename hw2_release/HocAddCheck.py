import numpy as np
import os, sys
from os import listdir
from itertools import product
from edge import *
from skimage import io
import matplotlib.pyplot as plt

# class B(Exception):
#     pass
# class C(B):
#     pass
# class D(C):
#     pass
# for cls in [C,B,D]:
#     try:
#         raise cls()
#     except B:
#         print("B")
#     except D:
#         print("D")
#     except C:
#         print("C")
#     # except B:
#     #     print("B")
# # Hoc = int(input())
# # print (" wrong theta value {} - should be on of the following [12,22,12,]".format(Hoc))
# while True:
#     try:
#         x = int(input("Please enter a number: "))
#         break
#     except ValueError:
#         print("Oops! That was no value number. Try again ...")
### Hoc check 2

# sigmas = [1.4,1.4,1.5]
# highs = [0.5,0.45,0.35]
# lows = [0.11,0.14,0.15]
# for sigma, high, low in product(sigmas, highs, lows):
#     print("sigma{}, high={}, low={}".format(sigma,high,low))
#     n_detected = 0.0
#     n_gt = 0.0
#     n_correct = 0.0
#     TP=0.0
#     FN=0.0
#     FP=0.0
#     for img_file in os.listdir('images/objects/'):
#         img = io.imread('images/objects/' + img_file, as_gray=True)
#         gt = io.imread('images/gt/' + img_file + '.gtf.pgm',as_gray=True)

#         mask = (gt != 5)    #'don't care region'
#         gt2 = (gt == 0)
#         #binary image of GT edges
#         gt3 = (gt == np.max(gt))
#         edges = canny(img, kernel_size = 5, sigma=sigma, high =high, low=low)
               
#         # compute FN - don't need this section
#         edges_invert = edges ^ True
#         FN += np.sum(edges_invert*gt3)
#         #compute FP - don't need this section
#         FP +=np.sum(edges * mask * gt2)
               
#         # compute TP
#         TP += np.sum(edges * gt3)
#         # compute TP + FP 
#         n_detected += np.sum(edges*mask)
#         # compute TP+FN
#         n_gt += np.sum(gt3)
#     n_correct = TP
#     p_total = n_correct/n_detected
#     r_total = n_correct/n_gt
#     f1 = 2*(p_total * r_total)/(p_total+r_total)
#     print('Total precision = {:.4f}, Total recall = {:.4f}'.format(p_total,r_total))
#     print('F1 score = {:.4f}'.format(f1))

##############Check3
# path = "images/objects/"
# dirs= os.listdir(path)
# for file in dirs:
#     img = io.imread(path+file,as_gray=True)
#     print(img)


# print("Hoc season! Hello world!")
# print('Hoc season! hello everyworld!')
# print('Enter sigma:')
# sigmas=input()
# print('Enter highs:')
# highs= input()
# print('Enter lows:')
# lows=input()

#Check4
# sigmas = [100,200,300]
# highs = [10,20,30]
# lows = [1,2,3]
# for a,b,c in product(sigmas,highs,lows):
#     print('results = {}'.format(a+b+c))

################### Check 5

# Load image
img = io.imread('road.jpg', as_gray=True)

# Run Canny edge detector
edges = canny(img, kernel_size=5, sigma=1.4, high=0.03, low=0.02)
H, W = img.shape
# Generate mask for ROI (Region of Interest)
mask = np.zeros((H, W))
for i in range(H):
    for j in range(W):
        if i > (H / W) * j and i > -(H / W) * j + H:
            mask[i, j] = 1

# Extract edges in ROI
roi = edges * mask 
# Perform Hough transform on the ROI
W, H = roi.shape
diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
rhos = np.linspace(-diag_len, diag_len, diag_len * 2 + 1)
thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
cos_t = np.cos(thetas)
sin_t = np.sin(thetas)
num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
ys, xs = np.nonzero(roi)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    # REFERENCE CODE
for i, j in zip(ys, xs):
    for idx in range(thetas.shape[0]):
        r = j * cos_t[idx] + i * sin_t[idx]
        accumulator[int(r + diag_len), idx] += 1
print(accumulator)
print(rhos)
acc = np.copy(accumulator) #Hoc add
#    return "accumulator", "rhos", "thetas"



# Coordinates for right lane
xs_right = []
ys_right = []

# Coordinates for left lane
xs_left = []
ys_left = []
for i in range(20):
    idx = np.argmax(acc)
    r_idx = idx // acc.shape[1]
    t_idx = idx % acc.shape[1]
    acc[r_idx, t_idx] = 0 # Zero out the max value in accumulator

    rho = rhos[r_idx]
    theta = thetas[t_idx]
    
    # Transform a point in Hough space to a line in xy-space.
    a = - (np.cos(theta)/np.sin(theta)) # slope of the line
    b = (rho/np.sin(theta)) # y-intersect of the line

    # Break if both right and left lanes are detected
    if xs_right and xs_left:
        break
    
    if a < 0: # Left lane
        if xs_left:
            continue
        xs = xs_left
        ys = ys_left
    else: # Right Lane
        if xs_right:
            continue
        xs = xs_right
        ys = ys_right

    for x in range(img.shape[1]):
        y = a * x + b
        if y > img.shape[0] * 0.6 and y < img.shape[0]:
            xs.append(x)
            ys.append(int(round(y)))
plt.imshow(img)
# plt.plot(xs_left, ys_left, linewidth=5.0)
plt.plot(xs_right, ys_right, linewidth=5.0)
plt.plot(xs_left, ys_left, linewidth=5.0)
plt.axis('off')
plt.show()