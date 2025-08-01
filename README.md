# No Entry Sign Detection

This project is a computer vision application for detecting "No Entry" signs in images. It utilizes a combination of three different techniques to achieve this:

*   **Viola-Jones Algorithm:** A robust object detection method that uses a cascade of classifiers trained on positive and negative image samples.
*   **Hough Transform for Circles:** A feature extraction technique used to identify circular shapes within an image.
*   **Template Matching:** A method for finding small parts of an image that match a template image.

The system can be run on a single image or a directory of images. It evaluates its performance using the True Positive Rate (TPR) and F1 Score metrics.

## Functionality

The `shape_detection.py` script is the main entry point of the application. It performs the following steps:

1.  **Image Loading:** Loads the input image(s) from the specified path or the `./No_entry/` directory.
2.  **Viola-Jones Detection:** Applies the pre-trained Viola-Jones classifier (`NoEntrycascade/cascade.xml`) to detect potential sign regions.
3.  **Hough Circle Transform:** Detects circles in the image using the Hough Transform.
4.  **Template Matching:** Matches a template of the "No Entry" sign (`no_entry.jpg`) against the image.
5.  **Candidate Fusion:** Combines the results from the different detectors and uses non-maximum suppression to eliminate duplicate detections.
6.  **Colour Filtering:** Filters the detected regions based on the presence of red, a dominant colour in "No Entry" signs.
7.  **Evaluation:** Compares the final detections against the ground truth data in `groundtruth.txt` and calculates the TPR and F1 Score.
8.  **Output:** Saves the images with the detected signs in the `./Output/` directory.

## File Descriptions

*   `shape_detection.py`: The main script that orchestrates the detection process.
*   `violaJones.py`: Implements the Viola-Jones object detection.
*   `houghSpace.py`: Implements the Hough Transform for circle detection.
*   `centralController.py`: A helper class for calculating evaluation metrics like IoU, TPR, and F1-score.
*   `groundtruth.txt`: Contains the ground truth bounding boxes for the images in the `No_entry` directory.
*   `NoEntrycascade/cascade.xml`: The pre-trained cascade classifier for the Viola-Jones algorithm.
*   `no_entry.jpg`: The template image used for template matching.
*   `negatives/`: A directory containing negative samples used for training the classifier.
*   `No_entry/`: A directory containing the test images with "No Entry" signs.
*   `Output/`: The directory where the output images with detected signs are saved.

## How to Run

To run the detection on the default set of images in the `./No_entry/` directory, simply execute the `shape_detection.py` script:

```bash
python shape_detection.py
```

To run the detection on a specific image, use the `--image` argument:

```bash
python shape_detection.py --image /path/to/your/image.jpg
```

## Dependencies

This project requires the following Python libraries:

*   **OpenCV:** `pip install opencv-python`
*   **NumPy:** `pip install numpy`
*   **Matplotlib:** `pip install matplotlib`