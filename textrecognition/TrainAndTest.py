from flask import Flask, render_template, request, redirect, url_for,jsonify
import time
import cv2
import numpy as np
import operator
import os

app = Flask(__name__)

MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

class ContourWithData():

    # member variables ############################################################################
    npaContour = None           
    boundingRect = None     
    intRectX = 0        
    intRectY = 0                
    intRectWidth = 0            
    intRectHeight = 0           
    fltArea = 0.0       

    def calculateRectTopLeftPointAndWidthAndHeight(self):               
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                          
        if self.fltArea < MIN_CONTOUR_AREA: return False        
        return True

@app.route('/',  methods=['GET'])
def template():
    return render_template(
        'index.html')


@app.route('/main',  methods=['GET', 'POST'])
def main():
    allContoursWithData = []                
    validContoursWithData = []              
    start_time = time.time()
    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)              
    except:
        print ("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32) 
    except:
        print ("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return
    # end try

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       

    kNearest =  cv2.ml.KNearest_create()                   

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    imgTestingNumbers = cv2.imread("2WQDWSA.JPG")          

    if imgTestingNumbers is None:                       
        print ("error: image not read from file \n\n")        
        os.system("pause")                                  
        return                                              
    # end if

    imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)       
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)    

                                                        
    imgThresh = cv2.adaptiveThreshold(imgBlurred,                           
                                      255,                                  
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       
                                      cv2.THRESH_BINARY_INV,                
                                      11,                                   
                                      2)                                    

    imgThreshCopy = imgThresh.copy()        

    _,npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,             
                                                 cv2.RETR_EXTERNAL,         
                                                 cv2.CHAIN_APPROX_SIMPLE)   
    for npaContour in npaContours:                             
        contourWithData = ContourWithData()                                             
        contourWithData.npaContour = npaContour                                        
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                   
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           
        allContoursWithData.append(contourWithData)                                     
    # end for

    for contourWithData in allContoursWithData:             
        if contourWithData.checkIfContourIsValid():             
            validContoursWithData.append(contourWithData)       
        # end if
    # end for

    validContoursWithData.sort(key = operator.attrgetter("intRectX"))         

    strFinalString = ""        

    for contourWithData in validContoursWithData:           
                                                
        cv2.rectangle(imgTestingNumbers,                                       
                      (contourWithData.intRectX, contourWithData.intRectY),    
                      (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),    
                      (255, 0, 0),              
                      2)                        

        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     
                           contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))           

        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))     

        npaROIResized = np.float32(npaROIResized)      

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)     

        strCurrentChar = str(chr(int(npaResults[0][0])))                                            

        strFinalString = strFinalString + strCurrentChar           
    # end for
    print (" %s seconds" %(time.time()-start_time))
    print ("\n" + strFinalString + "\n")                 
    index = 0

    cv2.imshow("imgTestingNumbers", imgTestingNumbers)      
    cv2.waitKey(0)                                         

    cv2.destroyAllWindows()            

    return render_template(
        'index.html', string=strFinalString)
    
    return ""


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
