class CentralController:

    def IoU(self, sign, ground):
        signX2 = sign[0] + sign[2]
        signY2 = sign[1] + sign[3]
        groundX2 = ground[0] + ground[2]
        groundY2 = ground[1] + ground[3]
        interX1 = max(sign[0], ground[0])
        interY1 = max(sign[1], ground[1])
        interX2 = min(signX2, groundX2)
        interY2 = min(signY2, groundY2)
        areaOfIntesection = 0
        if (interX1 <= interX2 and interY1 <= interY2):
            width = interX2 - interX1
            height = interY2 - interY1
            areaOfIntesection = width * height
        signArea = sign[2] * sign[3]
        groundArea = ground[2] * ground[3]
        areaOfUnion = signArea + groundArea - areaOfIntesection
        return areaOfIntesection / areaOfUnion

    def __findTruePositives(self, signs, grounds):
        total = 0
        groundsCheck = grounds.copy()
        for sign in signs:
            for ground in groundsCheck:
                if (self.IoU(sign, ground) > 0.5):
                    total += 1
                    groundsCheck.remove(ground)
                    break
        return total

    def getTPR(self, boxes, grounds):
        truePositives = self.__findTruePositives(boxes, grounds)
        falseNegatives = len(grounds) - truePositives
        return truePositives / (truePositives + falseNegatives)

    def getF1Score(self, boxes, grounds):
        truePositives = self.__findTruePositives(boxes, grounds)
        falseNegatives = len(grounds) - truePositives
        falsePositives = len(boxes) - truePositives
        return (2 * truePositives)/(2*truePositives +
                                    falseNegatives + falsePositives)
