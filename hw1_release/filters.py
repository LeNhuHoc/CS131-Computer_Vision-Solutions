"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    #Hoc
    # kernel=np.flip(kernel,axis=0)
    # kernel=np.flip(kernel,axis=1)

    temp_m = np.zeros((Hi+Hk-1, Wi+Wk-1))     # The result is a full matrix
    for i in range(Hi+Hk-1):
        for j in range(Wi+Wk-1):
            temp = 0
            # Generally speaking, the size of the convolution kernel is much smaller than the image size, and the convolution satisfies the commutative law. In order to speed up the calculation, you can use h*f instead of f*h for calculation
            for m in range(Hk):
                for n in range(Wk):
                    if ((i-m)>=0 and (i-m)<Hi and (j-n)>=0 and (j-n)<Wi):
                        temp += image[i-m][j-n] * kernel[m][n]
            temp_m[i][j] = temp
    # Cut out the same matrix (the output size is the same as the input)
    for i in range(Hi):
        for j in range(Wi):
            out[i][j] = temp_m[int(i+(Hk-1)/2)][int(j+(Wk-1)/2)]            

    return out
    pass
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.zeros((H+2*pad_height,W+2*pad_width))
    for i in range(H+2*pad_height):
        for j in range(W+2*pad_width):
            if ((i<pad_height) or (i>H+pad_height-1) or (j<pad_width) or (j>W+pad_width-1)):
                out[i,j]=out[i,j]
            else:
                out[i,j]=image[i-pad_height,j-pad_width]
    return out
    pass
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pad_height=Hk//2
    pad_width=Wk//2
    image_padding = zero_pad(image,pad_height,pad_width)
    kernel_flip=np.flip(np.flip(kernel,0),1)
    for i in range(Hi):
        for j in range(Wi):
            out[i,j]=np.sum(np.multiply(kernel_flip,image_padding[i:(i+Hk),j:(j+Wk)]))
    return out
    pass
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g=np.flip(np.flip(g,0),1)
    out = conv_fast(f,g)
    pass
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    mean_g=g.mean()
    g=g-mean_g
    out = cross_correlation(f,g)
    pass
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None

    # #Code version 1
    # ### YOUR CODE HERE
    # f=(f-np.mean(f)) / np.std(f)
    # g=(g-np.mean(g)) / np.std(g)
    # out = cross_correlation(f,g)
    # pass
    # ## END YOUR CODE

    # return out
    # #end


    # #Code version 2
    # ### YOUR CODE HERE
    # We don't support even kernel dimensions
    if g.shape[0] % 2 == 0:
        g = g[0:-1]
    if g.shape[1] % 2 == 0:
        g = g[:,0:-1]
    assert g.shape[0] % 2 == 1 and g.shape[1] % 2 == 1, "Even dimensions for filters is not allowed!"

    Hk, Wk = g.shape
    Hi, Wi = f.shape
    out = np.zeros((Hi, Wi))

    normalized_filter = (g - np.mean(g)) / np.std(g)
    assert np.mean(normalized_filter) < 1e-5, "g mean is {}, should be 0".format(np.mean(g))
    assert np.abs(np.std(normalized_filter) - 1) < 1e-5, "g std is {}, should be 1".format(np.std(g))

    delta_h = int((Hk - 1) / 2)
    delta_w = int((Wk - 1) / 2)
    for image_h in range(delta_h, Hi - delta_h):
        for image_w in range(delta_w, Wi - delta_w):
            image_patch =f[image_h - delta_h:image_h + delta_h + 1, image_w - delta_w:image_w + delta_w + 1]
            normalized_image_patch = (image_patch - np.mean(image_patch)) / np.std(image_patch)
            assert np.mean(normalized_image_patch) < 1e-5, "g mean is {}, should be 0".format(np.mean(g))
            assert np.abs(np.std(normalized_image_patch) - 1) < 1e-5, "g std is {}, should be 1".format(np.std(g))
            out[image_h][image_w] = np.sum(normalized_filter * normalized_image_patch)
    ### END YOUR CODE
    pass
    return out
