Major Color Extract using SWASA and S-CIELAB


Use: python main.py path/to/your/image.jpg --n_colors 5 --visualize --save output.png

Use as Lib in python:

from scielab import scielab_filter
from swasa import extract_dominant_colors
from PIL import Image
import numpy as np

'''Load Images'''
image_path = "path/to/your/image.jpg"
img = Image.open(image_path).convert('RGB')
img_array = np.array(img)

'''Apply for S-CIELAB filter'''
filtered_img = scielab_filter(img_array)

'''Exract Major Colors'''
colors, proportions = extract_dominant_colors(filtered_img, n_colors=5, max_iter=1000)

GPU Acceleration: See requirements.txt

Reference:

1. Zhang, X., & Wandell, B. A. (1996). A spatial extension of CIELAB for digital color image reproduction. SID International Symposium Digest of Technical Papers, 27, 731-734.
2. Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. Science, 220(4598), 671-680. 