import numpy as np
from scipy import signal
from scipy.ndimage import convolve1d
from skimage import color

def gauss(size, sigma):
    """
    Generate a one-dimensional Gaussian filter
    """
    x = np.arange(-size//2 + 1, size//2 + 1)
    g = np.exp(-(x**2) / (2 * sigma**2))
    return g / np.sum(g)

def separable_filters(samp_per_deg):
    """
    Generate separable filters for spatial filtering
    """
    # Luminance channel filter parameters
    s1 = 0.0283 * samp_per_deg
    s2 = 0.133 * samp_per_deg
    p1 = 0.7931
    p2 = 0.2069
    
    # Red-green channel filter parameters
    s3 = 0.0479 * samp_per_deg
    s4 = 0.1333 * samp_per_deg
    p3 = 0.7116
    p4 = 0.2884
    
    # Blue-yellow channel filter parameters
    s5 = 0.0670 * samp_per_deg
    s6 = 0.1333 * samp_per_deg
    p5 = 0.6064
    p6 = 0.3936
    
    # Create filters
    k1_size = int(6 * s2)
    if k1_size % 2 == 0:
        k1_size += 1
    k1 = p1 * gauss(k1_size, s1) + p2 * gauss(k1_size, s2)
    
    k2_size = int(6 * s4)
    if k2_size % 2 == 0:
        k2_size += 1
    k2 = p3 * gauss(k2_size, s3) + p4 * gauss(k2_size, s4)
    
    k3_size = int(6 * s6)
    if k3_size % 2 == 0:
        k3_size += 1
    k3 = p5 * gauss(k3_size, s5) + p6 * gauss(k3_size, s6)
    
    return k1, k2, k3

def separable_conv(image, kernel):
    """
    Perform 2D convolution using separable kernels
    """
    # First convolve along rows
    temp = convolve1d(image, kernel, axis=0, mode='reflect')
    # Then convolve along columns
    return convolve1d(temp, kernel, axis=1, mode='reflect')

def rgb_to_opponent(image):
    """
    Convert RGB image to opponent color space
    """
    # Convert RGB to XYZ
    xyz = color.rgb2xyz(image)
    
    # XYZ to opponent color space transformation matrix
    xyz_to_opp = np.array([
        [0.279, 0.72, 0.107],
        [0.449, -0.29, -0.077],
        [0.086, -0.59, 0.501]
    ])
    
    # Convert to opponent color space
    opp = np.zeros_like(xyz)
    for i in range(3):
        opp[..., i] = (xyz[..., 0] * xyz_to_opp[i, 0] + 
                       xyz[..., 1] * xyz_to_opp[i, 1] + 
                       xyz[..., 2] * xyz_to_opp[i, 2])
    
    return opp

def opponent_to_xyz(opponent):
    """
    Convert opponent color space back to XYZ space
    """
    # Opponent color space to XYZ transformation matrix (inverse of xyz_to_opp)
    opp_to_xyz = np.array([
        [1.0, 0.966, 0.336], 
        [1.0, -0.272, -0.636], 
        [1.0, -1.708, 1.092]
    ])
    
    # Convert back to XYZ
    xyz = np.zeros_like(opponent)
    for i in range(3):
        xyz[..., i] = (opponent[..., 0] * opp_to_xyz[i, 0] + 
                       opponent[..., 1] * opp_to_xyz[i, 1] + 
                       opponent[..., 2] * opp_to_xyz[i, 2])
    
    return xyz

def scielab_filter(image, samp_per_deg=23):
    """
    Implement S-CIELAB spatial filtering
    
    Parameters:
    image - RGB image, value range 0-255
    samp_per_deg - Samples per degree of visual angle, default is 23 (typical display)
    
    Returns:
    filtered_image - RGB image after S-CIELAB filtering
    """
    # Ensure image is float type and normalized
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # Convert to opponent color space
    opponent = rgb_to_opponent(image)
    
    # Generate filters
    k1, k2, k3 = separable_filters(samp_per_deg)
    
    # Filter each channel separately
    filtered_opp = np.zeros_like(opponent)
    filtered_opp[..., 0] = separable_conv(opponent[..., 0], k1)
    filtered_opp[..., 1] = separable_conv(opponent[..., 1], k2)
    filtered_opp[..., 2] = separable_conv(opponent[..., 2], k3)
    
    # Convert back to XYZ space
    filtered_xyz = opponent_to_xyz(filtered_opp)
    
    # Convert XYZ back to RGB
    filtered_rgb = color.xyz2rgb(filtered_xyz)
    
    # Ensure values are in 0-1 range
    filtered_rgb = np.clip(filtered_rgb, 0, 1)
    
    return filtered_rgb

def scielab_diff(image1, image2, samp_per_deg=23):
    """
    Calculate S-CIELAB difference between two images
    
    Parameters:
    image1, image2 - RGB images, value range 0-255
    samp_per_deg - Samples per degree of visual angle, default is 23 (typical display)
    
    Returns:
    diff_image - S-CIELAB difference image between the two images
    """
    # Ensure images are float type and normalized
    if image1.dtype == np.uint8:
        image1 = image1.astype(np.float32) / 255.0
    if image2.dtype == np.uint8:
        image2 = image2.astype(np.float32) / 255.0
    
    # Convert to opponent color space
    opponent1 = rgb_to_opponent(image1)
    opponent2 = rgb_to_opponent(image2)
    
    # Generate filters
    k1, k2, k3 = separable_filters(samp_per_deg)
    
    # Filter each channel separately
    filtered_opp1 = np.zeros_like(opponent1)
    filtered_opp1[..., 0] = separable_conv(opponent1[..., 0], k1)
    filtered_opp1[..., 1] = separable_conv(opponent1[..., 1], k2)
    filtered_opp1[..., 2] = separable_conv(opponent1[..., 2], k3)
    
    filtered_opp2 = np.zeros_like(opponent2)
    filtered_opp2[..., 0] = separable_conv(opponent2[..., 0], k1)
    filtered_opp2[..., 1] = separable_conv(opponent2[..., 1], k2)
    filtered_opp2[..., 2] = separable_conv(opponent2[..., 2], k3)
    
    # Convert back to XYZ space
    filtered_xyz1 = opponent_to_xyz(filtered_opp1)
    filtered_xyz2 = opponent_to_xyz(filtered_opp2)
    
    # Convert to LAB space
    lab1 = color.xyz2lab(filtered_xyz1)
    lab2 = color.xyz2lab(filtered_xyz2)
    
    # Calculate CIELAB difference
    delta_e = np.sqrt(np.sum((lab1 - lab2)**2, axis=2))
    
    return delta_e

def scielab_filter_with_mapping(image, samp_per_deg=23, n_jobs=None):
    """
    Implement S-CIELAB spatial filtering and create color mapping table, optimized with parallel processing
    
    Parameters:
    image - RGB image, value range 0-255
    samp_per_deg - Samples per degree of visual angle, default is 23 (typical display)
    n_jobs - Number of parallel jobs, default is CPU cores-1
    
    Returns:
    filtered_image - RGB image after S-CIELAB filtering
    color_mapping - Color mapping dictionary, keys are filtered color tuples, values are original color tuples
    """
    import multiprocessing
    from joblib import Parallel, delayed
    from collections import defaultdict
    
    # Set number of parallel jobs
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    
    # Ensure image is float type and normalized
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    
    # Flatten image for processing
    image_flat = image.reshape(-1, 3)
    
    # Convert to opponent color space
    opponent = rgb_to_opponent(image)
    
    # Generate filters
    k1, k2, k3 = separable_filters(samp_per_deg)
    
    # Filter each channel separately
    filtered_opp = np.zeros_like(opponent)
    filtered_opp[..., 0] = separable_conv(opponent[..., 0], k1)
    filtered_opp[..., 1] = separable_conv(opponent[..., 1], k2)
    filtered_opp[..., 2] = separable_conv(opponent[..., 2], k3)
    
    # Convert back to XYZ space
    filtered_xyz = opponent_to_xyz(filtered_opp)
    
    # Convert XYZ back to RGB
    filtered_rgb = color.xyz2rgb(filtered_xyz)
    
    # Ensure values are in 0-1 range
    filtered_rgb = np.clip(filtered_rgb, 0, 1)
    
    # Create color mapping - parallel optimized version
    filtered_flat = filtered_rgb.reshape(-1, 3)
    
    # Define batch processing function
    def process_batch(batch_indices, image_flat, filtered_flat):
        # Use local dictionary to accumulate colors
        local_color_sums = defaultdict(lambda: np.zeros(3, dtype=np.float32))
        local_color_counts = defaultdict(int)
        
        # Use float32 to reduce precision for faster processing, 6 decimal places is enough to distinguish most colors
        precision = 6
        
        for i in batch_indices:
            # Convert colors to tuples as keys
            filtered_array = filtered_flat[i]
            filtered_key = tuple(filtered_array)
            original_color = image_flat[i]
            
            local_color_sums[filtered_key] += original_color
            local_color_counts[filtered_key] += 1
            
        return local_color_sums, local_color_counts
    
    # Split data into batches
    batch_size = max(1, len(image_flat) // (n_jobs * 10))  # Each job processes multiple small batches to reduce overhead
    batch_indices = [range(i, min(i + batch_size, len(image_flat))) 
                   for i in range(0, len(image_flat), batch_size)]
    
    # Process each batch in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(indices, image_flat, filtered_flat)
        for indices in batch_indices
    )
    
    # Merge results
    merged_color_sums = defaultdict(lambda: np.zeros(3, dtype=np.float32))
    merged_color_counts = defaultdict(int)
    
    for local_sums, local_counts in results:
        for filtered_key, sum_color in local_sums.items():
            merged_color_sums[filtered_key] += sum_color
            merged_color_counts[filtered_key] += local_counts[filtered_key]
    
    # Create final mapping table, calculate averages
    color_mapping = {}
    for filtered_key, sum_color in merged_color_sums.items():
        count = merged_color_counts[filtered_key]
        color_mapping[filtered_key] = tuple(sum_color / count)
    
    return filtered_rgb, color_mapping