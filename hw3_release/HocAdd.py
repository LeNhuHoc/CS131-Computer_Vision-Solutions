import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve

from utils import pad, unpad, get_output_space, warp_image
from panorama import harris_corners
import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from skimage.io import imread
import matplotlib.pyplot as plt
from time import time
img = imread('sudoku.png', as_gray=True)
from collections import defaultdict
import numpy as np

ALPHABET_SIZE = 26

def check(count, k):
    for v in count.values():
        if v != k and v != 0:
            return False
    return True

def countSubstrings(s, k):
    total = 0
    for d in range(1, ALPHABET_SIZE + 1):
        size = d * k
        count = defaultdict(int)
        l = r = 0
        while r < len(s):
            count[s[r]] += 1
            # if window size exceed `size`, then fix left pointer and count
            if r - l + 1 > size:
                count[s[l]] -= 1
                l += 1
            # if window size is adequate then check and update count
            if r - l + 1 == size:
                total += check(count, k)
            r += 1  
    return total

def main():
    string1 = "1102021222"
    k1 = 2
    print(countSubstrings(string1, k1))        # output: 6
    
    string2 = "bacabcc"
    k2 = 2
    print(countSubstrings(string2, k2))        # output: 2
# def main():
#     print("Hello World!")

if __name__ == "__main__":
    main()


