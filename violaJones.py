import cv2


class Viola_Jones:
    def __init__(self, image, imgName):
        cascade_name = "./NoEntrycascade/cascade.xml"
        # 2. Load the Strong Classifier in a structure called `Cascade'
        self.model = cv2.CascadeClassifier()
        if not self.model.load(cascade_name):
            print('--(!)Error loading cascade model')
            exit(0)

        self.imgName = imgName
        self.image = image
        self.signs = []
        self.grounds = []
        self.__detectAndDisplay()

    def __detectAndDisplay(self):
        # 1. Prepare Image by turning it into Grayscale and normalising lighting
        frame_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        # 2. Perform Viola-Jones Object Detection
        signs = self.model.detectMultiScale(
            frame_gray, scaleFactor=1.1, minNeighbors=1, flags=0, minSize=(5, 5), maxSize=(300, 300))
        self.signs = signs

        # draw on ground truth signs
        self.__readGroundtruth()

    def __readGroundtruth(self, filename='groundtruth.txt'):
        # read bounding boxes as ground truth
        grounds = list()
        with open(filename) as f:
            # read each line in text file
            for line in f.readlines():
                content_list = line.split(",")
                img_name = content_list[0]
                # draw boxes with corresponding name
                if img_name == self.imgName:
                    x = float(content_list[1])
                    y = float(content_list[2])
                    width = float(content_list[3])
                    height = float(content_list[4])
                    start_point = (int(x), int(y))
                    end_point = (int(x+width), int(y+height))
                    colour = (0, 0, 255)
                    thickness = 2
                    self.image = cv2.rectangle(self.image, start_point,
                                               end_point, colour, thickness)
                    grounds.append([int(x), int(y), int(width), int(height)])
        self.grounds = grounds
        return grounds

    def getSigns(self):
        return self.signs

    def getGrounds(self):
        return self.grounds
