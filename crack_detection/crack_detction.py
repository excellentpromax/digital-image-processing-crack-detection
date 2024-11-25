import numpy as np
import cv2
from matplotlib import pyplot as plt

# List of image file names
image_files = ['2.png', '3.png', '5.jpg', '6.jpg']

# Function to process an image and return the processed images at each step
def process_image(image_path):
    # Read an image
    img = cv2.imread(image_path)

    # Convert into gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Image processing (smoothing)
    # Averaging
    blur = cv2.blur(gray, (3, 3))

    # Apply logarithmic transform
    img_log = (np.log(blur + 1) / (np.log(1 + np.max(blur)))) * 255

    # Specify the data type
    img_log = np.array(img_log, dtype=np.uint8)

    # Image smoothing: bilateral filter
    bilateral = cv2.bilateralFilter(img_log, 5, 75, 75)

    # Canny Edge Detection
    edges = cv2.Canny(bilateral, 100, 200)

    # Morphological Closing Operator
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Create feature detecting method
    orb = cv2.ORB_create(nfeatures=1500)

    # Make featured Image
    keypoints, descriptors = orb.detectAndCompute(closing, None)
    featuredImg = cv2.drawKeypoints(closing, keypoints, None)

    # Return all processed images
    return img, gray, blur, img_log, bilateral, edges, closing, featuredImg

# Process each image and store the results
all_results = []
for file in image_files:
    results = process_image(file)
    all_results.append(results)

# Function to display and save all the results for each image
def display_and_save_results(all_results, image_files):
    for i, results in enumerate(all_results):
        fig, axs = plt.subplots(nrows=1, ncols=8, figsize=(20, 5))
        axs[0].imshow(cv2.cvtColor(results[0], cv2.COLOR_BGR2RGB))
        axs[0].set_title('Original')
        axs[0].axis('off')

        axs[1].imshow(results[1], cmap='gray')
        axs[1].set_title('Gray')
        axs[1].axis('off')

        axs[2].imshow(results[2], cmap='gray')
        axs[2].set_title('Blur')
        axs[2].axis('off')

        axs[3].imshow(results[3], cmap='gray')
        axs[3].set_title('Log')
        axs[3].axis('off')

        axs[4].imshow(results[4], cmap='gray')
        axs[4].set_title('Bilateral')
        axs[4].axis('off')

        axs[5].imshow(results[5], cmap='gray')
        axs[5].set_title('Edges')
        axs[5].axis('off')

        axs[6].imshow(results[6], cmap='gray')
        axs[6].set_title('Closing')
        axs[6].axis('off')

        axs[7].imshow(results[7], cmap='gray')
        axs[7].set_title('Featured')
        axs[7].axis('off')

        plt.tight_layout()
        plt.savefig(f'Processed_{image_files[i]}')
        plt.show()

# Display and save the results for each image
display_and_save_results(all_results, image_files)