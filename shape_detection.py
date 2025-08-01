import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import re
from violaJones import Viola_Jones
from houghSpace import HoughSpace
from centralController import CentralController
import argparse


def nonMaximumSuppression(squares, T=0.3):
    squaresValid = []
    checked = np.empty((0, 4))
    i = 0
    while i < len(squares):
        square = squares[i]
        if np.any(np.all(checked == square, axis=1)):
            i += 1
            continue

        largestWithOverlap = square
        area = square[2] * square[3]
        j = i
        while j < len(squares):
            iOU = controller.IoU(square, squares[j])
            if iOU > T:
                size = squares[j][2] * squares[j][3]
                if size > area:
                    area = size
                    largestWithOverlap = squares[j]
                # Add square to the checked array
                checked = np.vstack((checked, squares[j]))
            j += 1

        i += 1
        squaresValid.append(largestWithOverlap)
    return squaresValid


def colourFiltering(img, box, thresh=0.3):
    # Extract the ROI
    roi = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]

    # Convert ROI to HSV color space
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define the color range for red (adjust values for your target)
    lower_red1 = np.array([0, 50, 50])    # Lower range for red (hue 0-10)
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])    # Lower range for red (hue 0-10)
    upper_red2 = np.array([180, 255, 255])

    # Create masks for the red color range
    mask1 = cv2.inRange(roi_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(roi_hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    redSum = np.sum(mask > 0)
    total = roi.shape[0] * roi.shape[1]
    percentage = redSum / total
    if (percentage > thresh):
        return True
    return False


def circCenter(img, boxes, circleBoxes, thresh=0.5):
    signs = []
    for box in boxes:
        for cir in circleBoxes:
            iOU = controller.IoU(box, cir)
            colourGood = colourFiltering(img, box)
            if (iOU > thresh and colourGood):
                signs.append(box)
                start_point = (box[0], box[1])
                end_point = (box[0] + box[2], box[1] + box[3])
                colour = (0, 255, 0)
                thickness = 2
                img = cv2.rectangle(img, start_point,
                                    end_point, colour, thickness)
                break
    return img, signs


def templateMatching(img, template="./no_entry.jpg"):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template, cv2.IMREAD_GRAYSCALE)
    template = cv2.resize(template, (220, 220))
    # Define scale range
    scales = np.linspace(0.1, 1.0, num=100)
    threshold = 0.5
    # Store all valid matches
    matches = []

    for scale in scales:
        # Resize the template
        resized_template = cv2.resize(
            template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        if resized_template.shape[0] > gray.shape[0] or resized_template.shape[1] > gray.shape[1]:
            continue

        # Perform template matching
        result = cv2.matchTemplate(
            gray, resized_template, cv2.TM_CCOEFF_NORMED)

        # Find all locations where the match score exceeds the threshold
        loc = np.where(result >= threshold)

        for pt in zip(*loc[::-1]):  # Reverse the order of coordinates to (x, y)
            matches.append((pt, scale, resized_template.shape))

    squares = []

    for match in matches:
        x0, y0 = match[0]
        w, h = match[2]
        squares.append([x0, y0, w, h])

    plt.show()

    return np.array(squares)


parser = argparse.ArgumentParser()
parser.add_argument('--image', dest='image', type=str, default="",
                    help='image path from scripts directory')
args = parser.parse_args()


if args.image != "":
    images = []
    imageNames = []
    try:
        img = cv2.imread(args.image)
        if not (type(img) is np.ndarray):
            print('Not image data')
            sys.exit(1)
        elif img is not None:
            images.append(img)
            name = args.image.split('/')[-1]
            imageNames.append(name.split('.')[0])
    except:
        print("Error occurred passing image path")
        sys.exit(1)

else:
    images = [None]*16
    imageNames = [None]*16
    # 1. Read Input Images
    folder = "./No_entry/"
    for image in os.listdir(folder):
        pos = re.findall(r'\d+', image)
        pos = int(pos[0])
        img = cv2.imread(os.path.join(folder, image))
        # ignore if image is not array.
        if image[:2] == "._":
            continue
        if not (type(img) is np.ndarray):
            print('Not image data')
            sys.exit(1)
        elif img is not None:
            images[pos] = img
            imageNames[pos] = image.split('.')[0]

controller = CentralController()

tprs = []
f1Scores = []
minRad = 6
# 3. Detect Signs and Display Result
for i in range(len(images)):
    img = images[i]
    imgName = imageNames[i]
    vJClass = Viola_Jones(img, imgName)
    houghCirc = HoughSpace(img, minRadii=minRad, maxRadii=107, increment=1)
    houghCirc.generateHough(T=100)
    template = templateMatching(img)
    boxes = np.concatenate((vJClass.getSigns(), template), axis=0)
    # houghCirc.drawCircles(T=15)
    signs = nonMaximumSuppression(boxes)
    img, signs = circCenter(img, signs, houghCirc.getBoxes(T=15))
    # 4. Save Result Image
    exists = os.path.exists("./Output")
    if not exists:
        os.makedirs("./Output")
    cv2.imwrite("./Output/detected"+str(i)+".jpg", img)
    tpr = controller.getTPR(signs, vJClass.getGrounds())
    f1Score = controller.getF1Score(signs, vJClass.getGrounds())
    tprs.append(tpr)
    f1Scores.append(f1Score)
    print("True Positive Rate for "+imgName+": "+str(round(tpr, 3)))
    print("f1 score for "+imgName+": "+str(round(f1Score, 3)))

if args.image == "":
    # Average the two scores
    avgTPR = sum(tprs) / len(tprs)
    avgf1 = sum(f1Scores) / len(f1Scores)
    print("Average True Positive Rate: "+str(round(avgTPR, 3)))
    print("Average f1 score: "+str(round(avgf1, 3)))
