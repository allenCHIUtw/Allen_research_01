import numpy as np
import cv2
from scipy.special import gamma
 
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
 
# ====================================================nqdm ==========================================
def estimate_ggd_params(vec):
    """Estimates GGD parameters (alpha, beta) using moment matching."""
    gam = np.arange(0.2, 10.0, 0.001)
    r_gam = (gamma(2/gam)**2) / (gamma(1/gam) * gamma(3/gam))
    
    mean_sq = np.mean(vec**2)
    if mean_sq == 0:
        return 0, 0
    r = (np.mean(np.abs(vec))**2) / mean_sq
    
    # Find closest ratio match
    pos = np.argmin(np.abs(r_gam - r))
    alpha = gam[pos]
    beta = np.sqrt(mean_sq * gamma(1/alpha) / gamma(3/alpha))
    return alpha, beta

def estimate_aggd_params(vec):
    """Estimates AGGD parameters (gamma, beta_l, beta_r, eta) using moment matching."""
    gam = np.arange(0.2, 10.0, 0.001)
    r_gam = (gamma(2/gam)**2) / (gamma(1/gam) * gamma(3/gam))
    
    left = vec[vec < 0]
    right = vec[vec > 0]
    
    # Simple fallbacks for empty splits
    if len(left) == 0: left = np.array([0.0])
    if len(right) == 0: right = np.array([0.0])
        
    m1 = np.mean(np.abs(vec))
    m2 = np.mean(vec**2)
    
    if m2 == 0:
        return 0, 0, 0, 0
        
    r = (m1**2) / m2
    pos = np.argmin(np.abs(r_gam - r))
    alpha = gam[pos]  # shape parameter parameter 'gamma' in paper
    
    gamma_1 = gamma(1/alpha)
    gamma_2 = gamma(2/alpha)
    gamma_3 = gamma(3/alpha)
    
    # Estimate left and right scale parameters
    r_hat = np.mean(left**2) / (np.mean(right**2) + 1e-6)
    beta_r = np.sqrt(m2 * (gamma_1 / gamma_3) / (1 + r_hat))
    beta_l = np.sqrt(r_hat) * beta_r
    
    # Compute mean (eta)
    eta = (beta_r - beta_l) * (gamma_2 / gamma_1)
    return alpha, beta_l, beta_r, eta

def extract_nss_features_for_patch(patch):
    """Extracts 18 spatial domain features from a single image patch scale."""
    features = []
    
    # 1. Fit GGD to MSC coefficients (yields 2 features: alpha, beta)
    alpha, beta = estimate_ggd_params(patch)
    features.extend([alpha, beta])
    
    # 2. Key directional products (Horizontal, Vertical, Main Diagonal, Anti-Diagonal)
    shifts = [
        (0, 1),  # Horizontal
        (1, 0),  # Vertical
        (1, 1),  # Main Diagonal
        (1, -1)  # Anti-Diagonal
    ]
    
    # Fit AGGD to each paired product (yields 4 features each -> 16 total)
    for shift_i, shift_j in shifts:
        shifted_patch = np.roll(patch, shift=shift_i, axis=0)
        shifted_patch = np.roll(shifted_patch, shift=shift_j, axis=1)
        
        # Calculate pixel-wise products
        product = patch * shifted_patch
        
        alpha_a, beta_l, beta_r, eta = estimate_aggd_params(product.flatten())
        features.extend([alpha_a, beta_l, beta_r, eta])
        
    return np.array(features)

def compute_image_features(image_gray):
    """Processes image across 2 downsampling scales to extract all 36 NIQE features."""
    img = image_gray.astype(np.float64)
    features_all_patches = []
    
    # Hardcoded block dimensions aligned with typical NIQE standards
    patch_size = 96 
    
    for scale in range(2):
        if scale > 0:
            # Low pass filter and downsample by a factor of 2
            img = cv2.resize(gaussian_filter(img, sigma=1.0), (img.shape[1]//2, img.shape[0]//2))
            
        # Local Mean Removal and Divisive Normalization (Equation 1, 2, 3)
        # Circularly symmetric Gaussian weighting function (sampled out to 3 std dev)
        mu = gaussian_filter(img, sigma=7/6) # ~ K=L=3 windowing mapping standard deviation
        mu_sq = mu * mu
        sigma = np.sqrt(np.abs(gaussian_filter(img * img, sigma=7/6) - mu_sq))
        struct_dis = (img - mu) / (sigma + 1)
        
        # Slice into non-overlapping P x P blocks
        h, w = struct_dis.shape
        scale_features = []
        
        for i in range(0, h - patch_size + 1, patch_size):
            for j in range(0, w - patch_size + 1, patch_size):
                patch = struct_dis[i:i+patch_size, j:j+patch_size]
                feat = extract_nss_features_for_patch(patch)
                scale_features.append(feat)
                
        if len(scale_features) > 0:
            if scale == 0:
                features_all_patches = scale_features
            else:
                # Combine Scale 1 and Scale 2 patch features side-by-side
                # To align with typical feature structures, ensure matching patch grids
                min_len = min(len(features_all_patches), len(scale_features))
                features_all_patches = [np.concatenate((features_all_patches[k], scale_features[k])) for k in range(min_len)]
                
    return np.array(features_all_patches)

def compute_niqe_score(test_image_path, model_mu, model_cov):
    """
    Computes the final quality metric score as the MVG distance between 
    the model statistics and the target image.
    """
    # Load and convert image to grayscale
    img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image from path: {test_image_path}")
        
    # Extract 36 total NSS features across patches
    patch_features = compute_image_features(img)
    
    # Fit target distorted image features to an MVG model
    test_mu = np.mean(patch_features, axis=0)
    test_cov = np.cov(patch_features, rowvar=False)
    
    # Compute the distance metric (Equation 10)
    # D = sqrt( (v1 - v2)^T * ((Sigma1 + Sigma2)/2)^-1 * (v1 - v2) )
    mean_diff = (model_mu - test_mu).reshape(-1, 1)
    avg_cov = (model_cov + test_cov) / 2.0
    
    # Calculate pinv to handle potential singular matrices safely
    inv_avg_cov = np.linalg.pinv(avg_cov)
    
    niqe_score = np.sqrt(np.dot(np.dot(mean_diff.T, inv_avg_cov), mean_diff))
    return float(niqe_score[0][0])