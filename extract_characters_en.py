'''
Function to draw red colour bounding box around character blobs.
1 parameter. Takes in image file path. 

Restrictions to fix:
! Does not consider the moment of characters.
! Does not denoise background of image. 
! Assumes uniform background color and lighting.
~ Line 43: figure size is hard coded. May not be optimal.
~ Line 47: character blob size paramenters is hard coded. Can be optimized.

'''

# Import necessary functions from scikit-image library
from skimage.io import imread, imshow, show
from skimage.transform import resize
from skimage.morphology import square, closing
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.filters import threshold_otsu
from matplotlib.cm import binary

def drawBBOX(filename='helloworld.jpg'):
    # Read in image, resize, and convert to greyscale
    im = imread(filename)
    im = resize(im, (500,500))
    im = rgb2gray(im)
    
    # Binarize image to black and white (black background with white blobs)
    thresh = threshold_otsu(im)
    bw = im < thresh
    
    # Morphological closing on binary image to fill in gaps
    im_close = closing(bw, square(7))
    
    # Label connected regions in the iamge array 
    label_img = label(im_close)
    
    # Clear objects connected to the label image border
    clean_border = clear_border(label_img)
    
    # Set up plot to prepare image with bbox
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(bw)
    
    # Character blob size parameters interested in
    min_height, max_height, min_width, max_width = (30, 600, 30, 600)
    
    # Counter for number of character blobs in image
    index = 0
    
    # For each character blob in the cleaned border labelled image
    for region in regionprops(clean_border):
        
        index = index + 1
        
        # pixels to the bounding box region
        min_row, min_col, max_row, max_col = region.bbox
        
        x = min_row
        y = min_col
        w = max_row
        h = max_col
        
        # calculate height and width from returned values from region.bbox
        region_height = max_row - min_row
        region_width = max_col - min_col
        
        # If the character blob meets these requirements, then draw rectangle shape
        if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
            
            # Specify the region of interest
            roi = bw[x:w, y:h]
            
            # Draw bounding box around region of interest
            rect_border = mpatches.Rectangle((y, x), h - y, w - x, edgecolor="red",
                                           linewidth=2, fill=False)
    
            ax.add_patch(rect_border)
            
            # Save each bounding box to a .png file with white background black blobs
            plt.imsave(filename+str(index)+".png", roi, cmap=binary)
    
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    
    print "Done."

