import numpy as np
import logging
import multiprocessing
from gpu_util import check_gpu_support, calculate_energy_gpu, update_labels_gpu
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SwasaProcessor:
    """SWASA processor class for encapsulating color clustering operations"""
    
    def __init__(self, use_gpu=False, n_jobs=None, disable_intel_gpu=False):
        """
        Initialize SWASA processor
        
        Parameters:
        use_gpu - Whether to use GPU acceleration
        n_jobs - Number of CPU parallel threads
        disable_intel_gpu - Whether to disable Intel GPU acceleration
        """
        self.use_gpu = use_gpu
        
        # Set number of parallel jobs
        if n_jobs is None and not use_gpu:
            self.n_jobs = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.n_jobs = n_jobs
            
        if self.n_jobs is not None and self.n_jobs > 1:
            if self.n_jobs > multiprocessing.cpu_count():
                logger.warning(f"Requested job count {self.n_jobs} exceeds available CPU count {multiprocessing.cpu_count()}, "
                             f"will use {multiprocessing.cpu_count()-1} cores")
                self.n_jobs = max(1, multiprocessing.cpu_count() - 1)
        
        self.disable_intel_gpu = disable_intel_gpu
        
        # Check GPU support only once
        if self.use_gpu:
            self.gpu_info = check_gpu_support()
            if not self.gpu_info['has_cuda'] and not self.gpu_info['has_intel']:
                logger.warning("No supported GPU detected, falling back to CPU computation")
                self.use_gpu = False
            elif self.disable_intel_gpu and not self.gpu_info['has_cuda'] and self.gpu_info['has_intel']:
                logger.info("Intel GPU has been disabled, falling back to CPU computation")
                self.use_gpu = False
                
        # Output computation device information
        if self.use_gpu:
            if self.gpu_info['has_cuda']:
                logger.info("Will use CUDA GPU acceleration for computation")
            elif self.gpu_info['has_intel']:
                logger.info("Will use Intel GPU acceleration for computation")
        elif self.n_jobs and self.n_jobs > 1:
            logger.info(f"Will use {self.n_jobs} CPU cores for parallel computation")
        else:
            logger.info("Will use single-threaded CPU computation")
    
    def calculate_energy(self, image_flat, centers, labels):
        """
        Calculate energy function
        
        Parameters:
        image_flat - Flattened image data
        centers - Cluster centers
        labels - Pixel labels
        
        Returns:
        total_error - Calculated energy value
        """
        if self.use_gpu:
            gpu_result = calculate_energy_gpu(image_flat, centers, labels, self.gpu_info)
            if gpu_result is not None:
                return gpu_result
        
        # CPU computation
        if self.n_jobs is not None and self.n_jobs > 1:
            def process_chunk(chunk_data, centers, labels_chunk):
                centers_for_pixels = centers[labels_chunk]
                squared_diff = np.sum((chunk_data - centers_for_pixels) ** 2, axis=1)
                return np.sum(squared_diff)
            
            chunk_size = max(1, len(image_flat) // self.n_jobs)
            chunks = [image_flat[i:i + chunk_size] for i in range(0, len(image_flat), chunk_size)]
            labels_chunks = [labels[i:i + chunk_size] for i in range(0, len(labels), chunk_size)]
            
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(process_chunk)(chunk, centers, labels_chunk)
                for chunk, labels_chunk in zip(chunks, labels_chunks)
            )
            
            return np.sum(results)
        else:
            centers_for_pixels = centers[labels]
            squared_diff = np.sum((image_flat - centers_for_pixels) ** 2, axis=1)
            return np.sum(squared_diff)
    
    def update_labels(self, image_flat, centers, n_colors):
        """
        Update pixel labels
        
        Parameters:
        image_flat - Flattened image data
        centers - Cluster centers
        n_colors - Number of colors
        
        Returns:
        labels - Updated label array
        """
        if self.use_gpu:
            gpu_result = update_labels_gpu(image_flat, centers, n_colors, self.gpu_info)
            if gpu_result is not None:
                return gpu_result
        
        # CPU computation
        if self.n_jobs is not None and self.n_jobs > 1:
            def process_chunk(chunk_data, centers):
                distances = np.zeros((len(chunk_data), n_colors))
                for j in range(n_colors):
                    diff = chunk_data - centers[j]
                    distances[:, j] = np.sum(diff * diff, axis=1)
                return np.argmin(distances, axis=1)
            
            chunk_size = max(1, len(image_flat) // self.n_jobs)
            chunks = [image_flat[i:i + chunk_size] for i in range(0, len(image_flat), chunk_size)]
            
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(process_chunk)(chunk, centers)
                for chunk in chunks
            )
            
            return np.concatenate(results)
        else:
            distances = np.zeros((len(image_flat), n_colors))
            for j in range(n_colors):
                diff = image_flat - centers[j]
                distances[:, j] = np.sum(diff * diff, axis=1)
            return np.argmin(distances, axis=1)
    
    def swasa_with_labels(self, image, n_colors=5, max_iter=1000, initial_temp=100000.0, 
                         cooling_rate=0.95, initial_step_size=0.1, min_step_size=0.001, 
                         adaptation_interval=50):
        """
        SWASA algorithm core implementation
        
        Parameters:
        image - Image data
        n_colors - Number of colors
        max_iter - Maximum iterations
        initial_temp - Initial temperature
        cooling_rate - Cooling rate
        initial_step_size - Initial step size
        min_step_size - Minimum step size
        adaptation_interval - Adaptation interval
        
        Returns:
        best_centers - Best cluster centers
        best_labels - Best labels
        proportions - Color proportions
        """
        # Ensure image values are in 0-1 range
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        
        # Initialize with K-means
        logger.info("Initializing cluster centers with K-means...")
        image_flat = image.reshape(-1, 3)
        try:
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_jobs=self.n_jobs)
            kmeans.fit(image_flat)
        except TypeError:
            logger.warning("Your scikit-learn version doesn't support n_jobs parameter for KMeans, using single-threaded initialization")
            kmeans = KMeans(n_clusters=n_colors, random_state=42)
            kmeans.fit(image_flat)
        
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        # Initialize best solution
        best_centers = centers.copy()
        best_labels = labels.copy()
        best_energy = self.calculate_energy(image_flat, centers, labels)
        
        # Annealing parameters
        temp = initial_temp
        step_size = initial_step_size
        accepted_count = 0
        
        logger.info(f"Starting SWASA algorithm, initial energy: {best_energy:.4f}")
        
        # Main loop
        for i in range(max_iter):
            # Generate candidate solution
            new_centers = centers.copy()
            idx = np.random.randint(0, len(centers))
            perturbation = np.random.uniform(-step_size, step_size, size=3) * temp
            new_centers[idx] = np.clip(centers[idx] + perturbation, 0, 1)
            
            # Update labels
            new_labels = self.update_labels(image_flat, new_centers, n_colors)
            
            # Calculate new energy
            new_energy = self.calculate_energy(image_flat, new_centers, new_labels)
            
            # Accept or reject new solution
            delta_e = new_energy - best_energy
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / temp):
                centers = new_centers
                labels = new_labels
                accepted_count += 1
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_centers = centers.copy()
                    best_labels = labels.copy()
            
            # Adjust step size
            if (i + 1) % adaptation_interval == 0:
                accepted_ratio = accepted_count / adaptation_interval
                step_size = max(step_size * (1 + 0.1 if accepted_ratio > 0.45 else -0.1), 
                              min_step_size)
                accepted_count = 0
                
                if (i + 1) % (adaptation_interval * 5) == 0:
                    logger.info(f"Iteration {i+1}/{max_iter}, temperature: {temp:.4f}, "
                              f"step size: {step_size:.4f}, energy: {best_energy:.4f}")
            
            # Cool down
            temp *= cooling_rate
        
        logger.info(f"SWASA algorithm completed, final energy: {best_energy:.4f}")
        
        # Calculate color proportions
        unique_labels, counts = np.unique(best_labels, return_counts=True)
        proportions = counts / len(best_labels)
        
        return best_centers, best_labels, proportions
    
    def extract_dominant_colors(self, image, n_colors=5, max_iter=1000):
        """
        Dominant color extraction function
        
        Parameters:
        image - RGB image, value range 0-255
        n_colors - Number of colors to extract
        max_iter - Maximum iterations for simulated annealing
        
        Returns:
        colors - Extracted color array, value range 0-255
        proportions - Proportion of each color
        """
        # Ensure image values are in 0-1 range
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        
        # Downsample processing
        height, width = image.shape[:2]
        max_dim = 200
        
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            new_height, new_width = int(height * scale), int(width * scale)
            
            indices_h = np.linspace(0, height-1, new_height, dtype=np.int32)
            indices_w = np.linspace(0, width-1, new_width, dtype=np.int32)
            
            resized_image = image[np.ix_(indices_h, indices_w)]
        else:
            resized_image = image
        
        # Apply SWASA algorithm
        centers, labels, proportions = self.swasa_with_labels(
            resized_image, n_colors=n_colors, max_iter=max_iter
        )
        
        # Prepare for sorting
        idx = np.argsort(proportions)[::-1]
        sorted_centers = centers[idx]
        sorted_proportions = proportions[idx]
        
        # Return filtered image colors (converted to 8-bit colors)
        result_centers = np.round(sorted_centers * 255).astype(np.uint8)
        
        return result_centers, sorted_proportions

# Keep original functions for compatibility, but implement using SwasaProcessor class
def swasa_with_labels(image, n_colors=5, max_iter=1000, initial_temp=100000.0, 
                     cooling_rate=0.95, initial_step_size=0.1, min_step_size=0.001, 
                     adaptation_interval=50, use_gpu=False, n_jobs=None):
    """SWASA algorithm wrapper function"""
    processor = SwasaProcessor(use_gpu=use_gpu, n_jobs=n_jobs)
    return processor.swasa_with_labels(
        image, n_colors, max_iter, initial_temp, 
        cooling_rate, initial_step_size, min_step_size, adaptation_interval
    )

def extract_dominant_colors(image, n_colors=5, max_iter=1000, 
                          use_gpu=False, n_jobs=None, disable_intel_gpu=False):
    """Dominant color extraction wrapper function"""
    processor = SwasaProcessor(use_gpu=use_gpu, n_jobs=n_jobs, disable_intel_gpu=disable_intel_gpu)
    return processor.extract_dominant_colors(image, n_colors, max_iter)