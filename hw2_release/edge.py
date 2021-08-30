"""
CS131 - Computer Vision: Foundations and Applications
Assignment 2
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/18/2017
Python Version: 3.5+
"""

import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    kernel=np.flip(np.flip(kernel,0),1)
    for i in range( Hi):
        for j in range(Wi):
            out[i,j] = np.sum(padded[i:i+Hk,j:j+Wk]*kernel)
    pass
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    k=size // 2
    for i in range(size):
        for j in range(size):
            kernel[i,j] = 1/(2*np.pi*sigma**2)*np.exp(-((i-k)**2+(j-k)**2)/(2*sigma**2))
    pass
    ### END YOUR CODE

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    filter_x = np.array([1,0,-1]).reshape(1,3)*1/2
    # filter_x=np.flip(np.flip(filter_x,0),1)
    out = conv(img,filter_x)
    pass
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    filter_y = np.array([1,0,-1]).reshape(3,1)*1/2
    # filter_y=np.flip(np.flip(filter_y,0),1)
    out = conv(img,filter_y)
    pass
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    G = np.sqrt(partial_x(img)**2+partial_y(img)**2)
    theta = (np.rad2deg(np.arctan2(partial_y(img),partial_x(img)))+360)%360
    pass
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    theta = theta % 360

    ### BEGIN YOUR CODE
    for i in range(1,H-1):
        for j in range(1,W-1):
            if theta[i,j]==0 or theta[i,j]==180:
                neighbors=[G[i,j+1],G[i,j-1]]
            elif theta[i,j]==45 or theta[i,j]==225:
                neighbors=[G[i-1,j+1],G[i+1,j-1]]
            elif theta[i,j]==90 or theta[i,j]==270:
                neighbors=[G[i-1,j],G[i+1,j]]
            elif theta[i,j]==135 or theta[i,j]==315:
                neighbors=[G[i-1,j-1],G[i+1,j+1]]
            else:
                raise RuntimeError("Wrong theta value {} - Should be one of the following [0,45,90,135,180,225,270,315]".format(theta[i,j]))
            if G[i,j]>=np.max(neighbors):
                out[i,j]=G[i,j]
            else:
                out[i,j]=0
    pass
    ### END YOUR CODE

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)

    ### YOUR CODE HERE
    strong_edges = img>high
    weak_edges = (img<high) & (img > low)
    pass
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    ### YOUR CODE HERE

    # weak_edges_new = np.zeros(strong_edges.shape)
    # for i in range(H):
    #     for j in range(W):
    #         if edges[i,j]==1:
    #             neighbors = get_neighbors(i,j,H,W)
    #             for k in range(len(neighbors)):
    #                 m,n = neighbors[k]
    #                 if weak_edges[m,n]==1:
    #                     weak_edges_new[m,n]=1
    # pass
    # edges = edges + weak_edges_new
    # # edges: numpy array of boolean of shape
    # edges = (edges==1)

    ### END YOUR CODE - TRUE
    ### REFERENCE CODE
    # Perform BFS!

    nodes_to_visit=[]
    visited_nodes = np.zeros_like(edges)
    nodes_to_visit.append((0,0))
    # While our nodes queue is not empty
    while len(nodes_to_visit) != 0:
            # Take the first element in the list
            curr_i, curr_j = nodes_to_visit.pop(0)

            # If we already visited this node - just continue
            if visited_nodes[curr_i, curr_j] == 1:
                continue

            # Mark node as visited
            visited_nodes[curr_i, curr_j] = 1

            neighors = get_neighbors(curr_i, curr_j, H, W)

            # Add neighbors
            for x,y in neighors:
                nodes_to_visit.append((x,y))

            # Check for adjacent edges
            adjacent_edges = False
            for x,y in neighors:
                adjacent_edges = edges[x, y] or adjacent_edges

            if weak_edges[curr_i,curr_j] and adjacent_edges:
                edges[curr_i,curr_j] = True
    ### END REFERENCE CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE

    kernel = gaussian_kernel(kernel_size,sigma)
    img_filter = conv(img,kernel)
    G,theta = gradient(img_filter)
    G_suppress = non_maximum_suppression(G,theta)
    strong_edges,weak_edges = double_thresholding(G_suppress,high,low)
    edge = link_edges(strong_edges,weak_edges)
    pass
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    # REFERENCE CODE
    for i, j in zip(ys, xs):
        for idx in range(thetas.shape[0]):
            r = j * cos_t[idx] + i * sin_t[idx]
            accumulator[int(r + diag_len), idx] += 1
    # END REFERENCE
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return accumulator, rhos, thetas