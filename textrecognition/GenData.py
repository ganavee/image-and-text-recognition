import sys
import numpy as np
import os
import cv2

#Global constants######
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
##############

def main():
    ImgTrainingNum = cv2.imread("training_chars.png")

    if ImgTrainingNum is None:
        print ("Error")
        os.system("pause")
        return

    ImgGray = cv2.cvtColor(ImgTrainingNum, cv2.COLOR_BGR2GRAY)
    ImgBlurr = cv2.GaussianBlur(ImgGray,(5,5), 0)
    ImgThresh = cv2.adaptiveThreshold(ImgBlurr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV,11,2)

    cv2.imshow("ImageThreshold", ImgThresh)

    ImgThreshCopy = ImgThresh.copy()

    _,npaContours, npaHierarchy = cv2.findContours(ImgThreshCopy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                                                                               
    npaFlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))#creating an empty array used to write the image data onto file
    intClassifications = []
    intValidCharacters = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

    for npaContour in npaContours:
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:
            [X, Y, W, H] = cv2.boundingRect(npaContour)
            cv2.rectangle(ImgTrainingNum,(X,Y),(X+W,Y+H),(0,255,0),2)
            ImgROI = ImgThresh[Y:Y+H, X:X+W]
            ImgROIResized = cv2.resize(ImgROI, (RESIZED_IMAGE_WIDTH,RESIZED_IMAGE_HEIGHT))

            cv2.imshow("ImageROI",ImgROI)
            cv2.imshow("ImageROIResized",ImgROIResized)
            cv2.imshow("training_numbers",ImgTrainingNum)

            k = cv2.waitKey(0)
            if k==27:
                sys.exit()
            elif k in intValidCharacters:
                intClassifications.append(k)
                # flatten image to 1d numpy array so we can write to file later
                npaFlattenedImage = ImgROIResized.reshape((1,RESIZED_IMAGE_WIDTH*RESIZED_IMAGE_HEIGHT))
                # add current flattened impage numpy array to list of flattened image numpy arrays
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)
                
    # convert classifications list of ints to numpy array of floats

    fltClassifications = np.array(intClassifications, np.float32)                   
    # flatten numpy array of floats to 1d so we can write to file later
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))  

    print ("\n\ntraining complete !!\n")

    np.savetxt("classifications.txt", npaClassifications)   
    np.savetxt("flattened_images.txt", npaFlattenedImages)          

    cv2.destroyAllWindows()            
    return


if __name__ == "__main__":
    main()
    
    
    
                   
    
    
