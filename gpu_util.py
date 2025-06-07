import numpy as np
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global GPU information
GPU_INFO = None

def check_gpu_support():
    """
    Check and initialize GPU support
    
    Returns:
        dict: Dictionary containing GPU support information
    """
    global GPU_INFO
    
    if GPU_INFO is not None:
        return GPU_INFO
        
    gpu_info = {
        'has_cuda': False,
        'has_intel': False,
        'has_device_context': False
    }
    
    # Check CUDA support
    try:
        import cupy as cp
        gpu_info['has_cuda'] = True
        logger.info("CUDA support detected")
    except ImportError:
        logger.info("CUDA support not detected")
    
    # Check Intel GPU support
    try:
        import dpctl
        import dpctl.tensor as dpt
        
        gpu_info['has_device_context'] = hasattr(dpctl, 'device_context')
        gpu_info['has_intel'] = any('gpu' in str(device.device_type).lower() 
                                  for device in dpctl.get_devices())
        
        if gpu_info['has_intel']:
            if gpu_info['has_device_context']:
                logger.info("Intel GPU support detected (using device_context API)")
            else:
                logger.info("Intel GPU support detected (using direct device access API)")
        else:
            logger.info("Intel GPU support not detected")
    except ImportError:
        logger.info("Intel oneAPI support not detected")
    
    GPU_INFO = gpu_info
    return gpu_info

def calculate_energy_gpu(image_flat, centers, labels, gpu_info):
    """
    Calculate energy using GPU if available
    
    Parameters:
        image_flat (ndarray): Flattened image array
        centers (ndarray): Cluster centers
        labels (ndarray): Pixel labels
        gpu_info (dict): Dictionary containing GPU support information
    
    Returns:
        float or None: Calculated energy value, or None if GPU calculation fails
    """
    if gpu_info['has_cuda']:
        import cupy as cp
        # CUDA GPU acceleration
        image_flat_gpu = cp.asarray(image_flat)
        centers_gpu = cp.asarray(centers)
        labels_gpu = cp.asarray(labels)
        
        centers_for_pixels = centers_gpu[labels_gpu]
        squared_diff = cp.sum((image_flat_gpu - centers_for_pixels) ** 2, axis=1)
        total_error = cp.sum(squared_diff).get()
        
        # Clean up memory
        del image_flat_gpu, centers_gpu, labels_gpu, centers_for_pixels, squared_diff
        cp.get_default_memory_pool().free_all_blocks()
        
        return total_error
    
    elif gpu_info['has_intel']:
        import dpctl
        import dpctl.tensor as dpt
        
        try:
            # Get GPU devices
            gpu_devices = [device for device in dpctl.get_devices() 
                         if 'gpu' in str(device.device_type).lower()]
            if not gpu_devices:
                raise RuntimeError("No available Intel GPU devices found")
            
            gpu_device = gpu_devices[0]
            
            # Calculate on GPU
            image_flat_gpu = dpt.asarray(image_flat, device=gpu_device)
            centers_gpu = dpt.asarray(centers, device=gpu_device)
            labels_gpu = dpt.asarray(labels, device=gpu_device)
            
            centers_for_pixels = centers_gpu[labels_gpu]
            squared_diff = dpt.sum((image_flat_gpu - centers_for_pixels) ** 2, axis=1)
            total_error = dpt.sum(squared_diff).asnumpy()
            
            return total_error
            
        except Exception as e:
            #logger.warning(f"Intel GPU energy calculation failed: {e}, falling back to CPU")
            return None
    
    return None  # Indicates GPU calculation not available

def update_labels_gpu(image_flat, centers, n_colors, gpu_info):
    """
    Update labels using GPU if available
    
    Parameters:
        image_flat (ndarray): Flattened image array
        centers (ndarray): Cluster centers
        n_colors (int): Number of colors
        gpu_info (dict): Dictionary containing GPU support information
    
    Returns:
        ndarray or None: Updated labels array, or None if GPU calculation fails
    """
    n_pixels = image_flat.shape[0]
    
    if gpu_info['has_cuda']:
        import cupy as cp
        # CUDA GPU acceleration
        image_flat_gpu = cp.asarray(image_flat)
        centers_gpu = cp.asarray(centers)
        
        distances = cp.zeros((n_pixels, n_colors))
        for j in range(n_colors):
            diff = image_flat_gpu - centers_gpu[j]
            distances[:, j] = cp.sum(diff * diff, axis=1)
        
        labels = cp.argmin(distances, axis=1).get()
        
        # Clean up memory
        del image_flat_gpu, centers_gpu, distances
        cp.get_default_memory_pool().free_all_blocks()
        
        return labels
    
    elif gpu_info['has_intel']:
        import dpctl
        import dpctl.tensor as dpt
        
        try:
            gpu_devices = [device for device in dpctl.get_devices() 
                         if 'gpu' in str(device.device_type).lower()]
            if not gpu_devices:
                raise RuntimeError("No available Intel GPU devices found")
            
            gpu_device = gpu_devices[0]
            
            image_flat_gpu = dpt.asarray(image_flat, device=gpu_device)
            centers_gpu = dpt.asarray(centers, device=gpu_device)
            
            distances = dpt.zeros((n_pixels, n_colors), device=gpu_device)
            for j in range(n_colors):
                diff = image_flat_gpu - centers_gpu[j]
                distances[:, j] = dpt.sum(diff * diff, axis=1)
            
            labels = dpt.argmin(distances, axis=1).asnumpy()
            return labels
            
        except Exception as e:
            #logger.warning(f"Intel GPU label update failed: {e}, falling back to CPU")
            return None
    
    return None  # Indicates GPU calculation not available