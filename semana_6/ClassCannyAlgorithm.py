import numpy as np
from scipy.ndimage import convolve
from scipy import ndimage


class canny:
    def __init__(self, img, sigma, size):
        """
        Initialized the class canny wiht the two hyperparameter:
        - img: image gray scale
        - sigma: hyperparameter to control the extraction of line
        - size: kernel size
        """
        self.img = img
        self.sigma = sigma
        self.size = size

    # Noise reduction
    def gaussian_kernel(self):
        """Noise reduction applying a Gaussian Kenl

        Returns:
            covolve image : Image convolved by using a specific kernel
        """
        size = int(self.size) // 2
        x, y = np.mgrid[-size : size + 1, -size : size + 1]
        normal = 1 / (2.0 * np.pi * self.sigma**2)
        g = np.exp(-((x**2 + y**2) / (2.0 * self.sigma**2))) * normal
        return g

    def sobel_filters(self):
        """Solber filter allows the application a image convolution to extract the gradiente calculation

        Returns:
            tuple: Image with the gradient intensity
        """
        img_conv = convolve(self.img, self.gaussian_kernel())
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        ix = ndimage.filters.convolve(img_conv, kx)
        iy = ndimage.filters.convolve(img_conv, ky)

        g_upper = np.hypot(ix, iy)
        g_upper = g_upper / g_upper.max() * 255
        theta = np.arctan2(iy, ix)

        return (g_upper, theta)

    def non_max_suppression(self):
        """The principle is simple: the algorithm goes through all the points on
        the gradient intensity matrix and finds the pixels with the maximum value
        in the edge directions

        Returns:
            image: Not max expression of the image
        """
        image, D = self.sobel_filters()
        M, N = image.shape
        Z = np.zeros((M, N), dtype=np.int32)
        angle = D * 180.0 / np.pi
        angle[angle < 0] += 180

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                try:
                    q = 255
                    r = 255

                    # angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = image[i, j + 1]
                        r = image[i, j - 1]
                    # angle 45
                    elif 22.5 <= angle[i, j] < 67.5:
                        q = image[i + 1, j - 1]
                        r = image[i - 1, j + 1]
                    # angle 90
                    elif 67.5 <= angle[i, j] < 112.5:
                        q = image[i + 1, j]
                        r = image[i - 1, j]
                    # angle 135
                    elif 112.5 <= angle[i, j] < 157.5:
                        q = image[i - 1, j - 1]
                        r = image[i + 1, j + 1]

                    if (image[i, j] >= q) and (image[i, j] >= r):
                        Z[i, j] = image[i, j]
                    else:
                        Z[i, j] = 0

                except IndexError as e:
                    pass

        return Z

    def threshold(self, lowThresholdRatio=0.05, highThresholdRatio=0.09):
        """
        - Strong pixels are pixels that have an intensity so high that we are sure they contribute to the final edge
        - Weak pixels are pixels that have an intensity value that is not enough to be considered as strong ones,
            but yet not small enough to be considered as non-relevant for the edge detection.
        - Other pixels are considered as non-relevant for the edge
        Returns:
            tuple (image):
        """

        image = self.non_max_suppression()
        highThreshold = image.max() * highThresholdRatio
        lowThreshold = highThreshold * lowThresholdRatio

        M, N = image.shape
        res = np.zeros((M, N), dtype=np.int32)

        weak = np.int32(25)
        strong = np.int32(255)

        strong_i, strong_j = np.where(image >= highThreshold)
        zeros_i, zeros_j = np.where(image < lowThreshold)

        weak_i, weak_j = np.where((image <= highThreshold) & (image >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return res

    def hysteresis(self, weak=75, strong=255):
        """The same step is the integration with hysteresis process to extract the lines

        Returns:
            list: Image with the principal lines
        """

        image = self.threshold()
        M, N = image.shape
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if image[i, j] == weak:
                    try:
                        if (
                            (image[i + 1, j - 1] == strong)
                            or (image[i + 1, j] == strong)
                            or (image[i + 1, j + 1] == strong)
                            or (image[i, j - 1] == strong)
                            or (image[i, j + 1] == strong)
                            or (image[i - 1, j - 1] == strong)
                            or (image[i - 1, j] == strong)
                            or (image[i - 1, j + 1] == strong)
                        ):
                            image[i, j] = strong
                        else:
                            image[i, j] = 0
                    except IndexError as e:
                        pass
        return image
