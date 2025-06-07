import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend to avoid Tkinter errors
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from PIL import Image
import time
import multiprocessing
from scielab import scielab_filter_with_mapping
from swasa import SwasaProcessor

def load_image(image_path):
    """
    Load image and convert to RGB format
    """
    img = Image.open(image_path)
    img = img.convert('RGB')
    return np.array(img)

def restore_original_colors(filtered_colors, color_mapping):
    """
    Restore filtered colors to original colors
    
    Parameters:
    filtered_colors - Filtered color array, range 0-255
    color_mapping - Color mapping table, keys are filtered color tuples, values are original color tuples
    
    Returns:
    original_colors - Restored original color array, range 0-255
    """
    # Convert to 0-1 range float
    filtered_colors_float = filtered_colors.astype(np.float32) / 255.0
    
    # Create original color array
    original_colors = np.zeros_like(filtered_colors_float)
    
    # Extract color key-values from mapping table
    from scipy.spatial import cKDTree
    
    # Build KD tree for fast nearest neighbor search
    start_time = time.time()
    mapping_keys = np.array(list(color_mapping.keys()))
    mapping_values = np.array(list(color_mapping.values()))
    kdtree = cKDTree(mapping_keys)
    
    # Batch query nearest neighbors
    distances, indices = kdtree.query(filtered_colors_float, k=1)
    
    # Use indices to directly get corresponding original colors
    for i in range(len(filtered_colors_float)):
        original_colors[i] = mapping_values[indices[i]]
    
    # Convert back to 0-255 range
    return np.round(original_colors * 255).astype(np.uint8)

def show_colors(colors, proportions=None):
    """
    Display extracted dominant colors
    """
    n_colors = len(colors)
    fig, ax = plt.subplots(1, n_colors, figsize=(n_colors * 2, 2))
    
    for i, color in enumerate(colors):
        rgb = np.array(color).reshape(1, 1, 3)
        if n_colors > 1:
            ax[i].imshow(rgb)
            if proportions is not None:
                ax[i].set_title(f"{proportions[i]:.2%}")
            ax[i].axis('off')
        else:
            ax.imshow(rgb)
            if proportions is not None:
                ax.set_title(f"{proportions[i]:.2%}")
            ax.axis('off')
    
    plt.tight_layout()
    return fig

def show_comparison(original_colors, filtered_colors, proportions=None):
    """
    Compare and display original and filtered colors
    """
    n_colors = len(original_colors)
    if len(filtered_colors) != n_colors:
        raise ValueError("Original colors and filtered colors must have the same count")
    
    fig, axes = plt.subplots(2, n_colors, figsize=(n_colors * 2, 4))
    
    # Titles
    axes[0, n_colors//2].set_title('Original Image Colors', fontsize=12)
    axes[1, n_colors//2].set_title('S-CIELAB Filtered Colors', fontsize=12)
    
    for i in range(n_colors):
        # Display original colors
        original_rgb = np.array(original_colors[i]).reshape(1, 1, 3)
        axes[0, i].imshow(original_rgb)
        if proportions is not None:
            axes[0, i].set_title(f"{proportions[i]:.2%}")
        axes[0, i].axis('off')
        
        # Display filtered colors
        filtered_rgb = np.array(filtered_colors[i]).reshape(1, 1, 3)
        axes[1, i].imshow(filtered_rgb)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    """
    Main program entry
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract dominant colors from image')
    parser.add_argument('image_path', type=str, help='Input image path')
    parser.add_argument('--n_colors', type=int, default=5, help='Number of dominant colors to extract')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum iterations for simulated annealing')
    parser.add_argument('--save', type=str, default=None, help='Path to save results')
    parser.add_argument('--visualize', action='store_true', help='Whether to display results')
    parser.add_argument('--use_gpu', action='store_true', help='Whether to use GPU acceleration (NVIDIA requires CuPy, Intel requires oneAPI)')
    parser.add_argument('--n_jobs', type=int, default=None, 
                        help=f'Number of CPU parallel threads (default uses {max(1, multiprocessing.cpu_count()-1)} threads)')
    parser.add_argument('--preview', action='store_true', help='Whether to display S-CIELAB filtered preview image')
    parser.add_argument('--disable_intel_gpu', action='store_true', help='Disable Intel GPU acceleration (use when compatibility issues occur)')
    parser.add_argument('--use_original_colors', action='store_true', help='Whether to use original image colors (instead of S-CIELAB filtered colors)')
    parser.add_argument('--compare_colors', action='store_true', help='Compare and display original and filtered colors')
    
    args = parser.parse_args()
    
    # Load image
    print(f"Loading image: {args.image_path}")
    start_time = time.time()
    original_image = load_image(args.image_path)
    
    # Apply S-CIELAB filter and create color mapping table
    print("Applying S-CIELAB filter and creating color mapping table...")
    filtered_image, color_mapping = scielab_filter_with_mapping(original_image)
    print(f"Created color mapping table with {len(color_mapping)} color pair mappings")
    
    # Display filtered preview
    if args.preview:
        preview_file = 'filtered_preview.png'
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(filtered_image)
        plt.title('S-CIELAB Filtered')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(preview_file)
        plt.close()
        print(f"Filter preview saved to: {preview_file}")
    
    # Extract dominant colors using SWASA
    print(f"Extracting {args.n_colors} dominant colors from S-CIELAB processed image using SWASA...")
    extract_start = time.time()
    
    # Create SwasaProcessor instance
    swasa_processor = SwasaProcessor(
        use_gpu=args.use_gpu,
        n_jobs=args.n_jobs,
        disable_intel_gpu=args.disable_intel_gpu
    )
    
    # Extract colors from processed image
    filtered_colors, proportions = swasa_processor.extract_dominant_colors(
        filtered_image, 
        n_colors=args.n_colors, 
        max_iter=args.max_iter
    )
    
    if args.compare_colors:
        # Restore filtered colors to original colors
        print("Restoring original colors...")
        original_colors = restore_original_colors(filtered_colors, color_mapping)
    else:
        if args.use_original_colors:
            # Restore to original colors using color mapping table
            print("Restoring original colors...")
            colors = restore_original_colors(filtered_colors, color_mapping)
        else:
            # Use processed colors directly
            colors = filtered_colors
    
    extract_end = time.time()
    
    end_time = time.time()
    total_time = end_time - start_time
    extract_time = extract_end - extract_start
    
    print(f"Processing complete, total time: {total_time:.2f} seconds, color extraction time: {extract_time:.2f} seconds")
    
    # Display results
    if args.compare_colors:
        print(f"Comparison of dominant colors between original image and S-CIELAB processed image:")
        for i, (orig_color, filt_color, prop) in enumerate(zip(original_colors, filtered_colors, proportions)):
            print(f"Dominant color {i+1}: Original RGB={orig_color}, Processed RGB={filt_color}, Proportion={prop:.2%}")
        
        # Create comparison visualization
        fig = show_comparison(original_colors, filtered_colors, proportions)
        
        # Save results
        if args.save:
            fig.savefig(args.save)
            print(f"Comparison results saved to: {args.save}")
        else:
            temp_file = 'S-CIELAB_color_comparison.png'
            fig.savefig(temp_file)
            print(f"Comparison results saved to: {temp_file}")
        
        plt.close(fig)
    else:
        # Display single result
        for i, (color, prop) in enumerate(zip(colors, proportions)):
            color_type = "Original" if args.use_original_colors else "S-CIELAB processed"
            print(f"Dominant color {i+1}: {color_type} RGB={color}, Proportion={prop:.2%}")
        
        # Create color visualization
        fig = show_colors(colors, proportions)
        
        # Save results
        if args.save:
            fig.savefig(args.save)
            print(f"Results saved to: {args.save}")
        elif args.visualize:
            # If no save path specified but visualization requested, save to temp file
            temp_file = 'S-CIELAB_color_palette.png'
            fig.savefig(temp_file)
            print(f"Due to non-interactive backend, results saved to: {temp_file}")
            print("Please use an image viewer to open this file to view results")
            
        plt.close(fig)

if __name__ == "__main__":
    main() 