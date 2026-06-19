import numpy as np
import cv2
 
from scipy.ndimage import gaussian_filter

def ssim_nd(X, Y ,data_range=255.0 ,K1=0.01,K2=0.03, sigma=1.5):
    if X.shape != Y.shape:
        raise ValueError("Input arrays must have the same shape.")
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # 2. Local Means (mu) using N-Dimensional Gaussian Filter
    # 'truncate=3.5' makes the window size ~ 2 * truncate * sigma + 1 
    # (e.g., for sigma=1.5, window is ~11 pixels wide along every dimension)
    mu1 = gaussian_filter(X, sigma=sigma, truncate=3.5)
    mu2 = gaussian_filter(Y, sigma=sigma, truncate=3.5)

    # Squares and products of means
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # 3. Local Variances (sigma^2) and Covariances (sigma_xy)
    # Calculated using the identity: Var(X) = E[X^2] - (E[X])^2
    sigma1_sq = gaussian_filter(X ** 2, sigma=sigma, truncate=3.5) - mu1_sq
    sigma2_sq = gaussian_filter(Y ** 2, sigma=sigma, truncate=3.5) - mu2_sq
    sigma12 = gaussian_filter(X * Y, sigma=sigma, truncate=3.5) - mu1_mu2

    # 4. Compute the SSIM map
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / den

    # 5. Return the mean SSIM (MSSIM) across the whole signal
    return np.mean(ssim_map)
 