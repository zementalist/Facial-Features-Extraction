import cv2
import dlib
from imutils import face_utils
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib import transforms
import math
from geometry import slope



def getAllowedColorRange(avgSkinColor):
    # Function to determine the color range allowed to move landmarks points through image
    # Dark skin
    if (avgSkinColor < 100):
        colorRange = (avgSkinColor-35, avgSkinColor+50)
    # Somehow dark skin
    elif(avgSkinColor <= 130): 
        colorRange = (avgSkinColor-30, avgSkinColor+30)
    # Normal skin color (tends to dark)
    elif(avgSkinColor <= 160):
        colorRange = (avgSkinColor-40, avgSkinColor+40) 
    # Normal skin color 
    elif(avgSkinColor < 180):
        colorRange = (avgSkinColor-50, avgSkinColor+50)
    # Normal skin color (tends to white)
    elif(avgSkinColor < 210):
        colorRange = (avgSkinColor-50, avgSkinColor+30) 
    # white skin color
    elif (avgSkinColor < 230):
        colorRange = (avgSkinColor-40, avgSkinColor+20)
    # Abnormal white skin color
    else:
        colorRange = (avgSkinColor-30, avgSkinColor+15)
        
    return colorRange

def moveUp(grayscale_image, point, avgSkinColor, foreheadHeight):
    # Function to move landmarks points, based on skincolor
    # Get color range & current color where the point is located in image
    steps = 5
    portionOfOriginalPointY = 0.275
    originalPoint = np.copy(point)
    colorRange = getAllowedColorRange(avgSkinColor)
    currentPixelColor = grayscale_image.item(point[1],point[0])
    
    # move the landmark point up until a strong change of color happen (outside color range)
    while currentPixelColor > colorRange[0] and currentPixelColor < colorRange[1]:
        # If point is going out of image boundary
        if point[1] < 0:
            # Get back to original point location, with a little bit higher
            point[1] = originalPoint[1] - (originalPoint[1] * portionOfOriginalPointY)
            break
        # move up (N steps) pixels & get the color
        point[1] = point[1] - steps
        currentPixelColor = grayscale_image.item(point[1],point[0])
    # if the pixel is moved too high than expected (3/4 forehead height): keep close to original
    if abs( originalPoint[1] - point[1] ) > ( foreheadHeight * 0.75 ):
        point[1] = originalPoint[1] - (originalPoint[1] * portionOfOriginalPointY)
    return point

def clearForehead(forehead, avgSkinColor):
    # Function to detect if the forehead is clear or covered with hair (it corrupts the enhancement of landmarks points)
    clarityThreshold = 85
    colorRange = getAllowedColorRange(avgSkinColor)
    # Check if most of the forehead is the same as skin color
    regionOK = np.logical_and(forehead > colorRange[0] , forehead < colorRange[1])
    try:
        percentage = (np.count_nonzero(regionOK) / forehead.size) * 100
    except:
        return False
    isClear = True if percentage >= clarityThreshold else False
    return isClear


def facial_landmarks(image, eyeOnlyMode=False, allowEnhancement=False):
    # Function to perform facial landmark detection on the whole face

    # Use dlib 68 & 81 to predict landmarks points coordinates
    detector = dlib.get_frontal_face_detector()
    predictor68 = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')
    predictor81 = dlib.shape_predictor('../shape_predictor_81_face_landmarks.dat')
    
    # Grayscale image
    try:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        grayscale_image = image
    
    # array of rectangles surrounding faces detected
    rectangles = detector(grayscale_image, 1)

    # If at least one face is detected   
    if len(rectangles) > 0:
        # Get 68 landmark points
        faceLandmarks = predictor68(grayscale_image, rectangles[0])
        faceLandmarks = face_utils.shape_to_np(faceLandmarks)
        
        if eyeOnlyMode:
            # Return eye points to perform a calculated rotation
            return np.array([faceLandmarks[39], faceLandmarks[42]])
        
        # Get 81 landmark points
        foreheadLandmarks = predictor81(grayscale_image, rectangles[0])
        foreheadLandmarks = face_utils.shape_to_np(foreheadLandmarks)
        
        # Get 68 point from -68- predictor (higher accuracy) + forehead from -81- predictor
        fullFacePoints = np.concatenate((faceLandmarks, foreheadLandmarks[68:]))
        
        # Get forehead region & height to perform simple improvement
        x,y,x2,y2 = (fullFacePoints[69,0]-10, fullFacePoints[68,1], fullFacePoints[80,0]+10, fullFacePoints[23, 1])
        foreheadRegion = grayscale_image[y:y2,x:x2]
        foreheadHeight = foreheadRegion.shape[0]
        
        if allowEnhancement:
            # Perform progressive quality improvement
            # Get nose region to get average skin color
            x,y,x2,y2 = (fullFacePoints[28,0]-5, fullFacePoints[28,1], fullFacePoints[28,0]+5, fullFacePoints[30,1])
            noseRegion = grayscale_image[y:y2, x:x2]
            avgSkinColor = np.average(noseRegion[:,:])
            
            # Check if forehead is clear -> perform heuristic based enhancement
            forehead_is_clear = clearForehead(foreheadRegion, avgSkinColor)
            originalPoints = fullFacePoints[[69,70,71,73,80]]
            
            if forehead_is_clear:
                avgSkinColor = np.average(foreheadRegion)
                
                # Modify some points for more accuracy
                # Point[68] will be center between lower-lip & chin
                distance = int((fullFacePoints[8,1]-fullFacePoints[57,1]) / 2)
                fullFacePoints[68] = np.array([fullFacePoints[8,0], fullFacePoints[8,1]-distance])
                
                # Enhance points locations
                enhancedPoints = np.array([moveUp(grayscale_image, orgPoint, avgSkinColor, foreheadHeight) for orgPoint in originalPoints])

                # Assign original points to enhanced points (some maybe the same)
                fullFacePoints[[69,70,71,73,80]] = enhancedPoints  
                
                # Adjust points to fix any corruptions
                fullFacePoints[[69,70,71,73,80]] = adjustPoints(enhancedPoints, fullFacePoints[76], fullFacePoints[79])

                #Prepare point[72] for center of forehead
                distance = (fullFacePoints[22,0] - fullFacePoints[21,0]) / 2
                distanceY = (fullFacePoints[21,1] - fullFacePoints[71,1]) / 2
                fullFacePoints[72] = np.array([fullFacePoints[21,0] + distance, fullFacePoints[21,1]-distanceY])
                
                # Point[74] sometimes have a fixed corruption, this line helps :)
                fullFacePoints[74,0] -= foreheadHeight * 0.1 # Arbitery heurestic
                
            else:
                # If forehead isn't clear -> fix points with very simple heuristics
                fullFacePoints[70,1] -= foreheadHeight * 0.2
                fullFacePoints[71,1] -= foreheadHeight * 0.3
                fullFacePoints[80,1] -= foreheadHeight * 0.2
    
        else:
            # If Enhancement is False -> do the simple enhancement, better quality + low performance :)
            fullFacePoints[70,1] -= foreheadHeight * 0.2
            fullFacePoints[71,1] -= foreheadHeight * 0.3
            fullFacePoints[80,1] -= foreheadHeight * 0.2
            pass
        
        return fullFacePoints
    # No faces found
    else:
        return None


def adjustPoints(points, leftSidePoint, rightSidePoint):
    # Function to adjust landmarks points of the forehead & fix corruptions of improvement
    
    # Use shape_predictor_81 as a reference for points indexes:
    # points = [69,70,71,73,80]
    # LeftSidePoint = 76  |  rightSidePoint = 79
    
    slopes = []
    slopeThreshold = 0.4 # slope > 0.4 = corruption -> fix
    totalSlopeThreshold = 1 # sum of slopes > 1 = corruption -> fix
    leftPoint = points[0]
    rightPoint = points[3]
    criticalLeftPoint = points[1]
    criticalRightPoint = points[4]
    
    # if any point is higher than a (accurate located point) -> fix
    if leftPoint[1] < criticalLeftPoint[1] :
        points[0,1] = np.average([criticalLeftPoint[1], leftSidePoint[1]])
    if rightPoint[1] < criticalRightPoint[1]:
        points[3,1] = np.average([criticalRightPoint[1], rightSidePoint[1]])
    
    # Collect some slopes of the usually corrupted points
    slopes.append(slope(points[1], points[2], True))
    slopes.append(slope(points[2], points[4], True))
    
    # Calculate slope differences & sum
    difference = abs(np.diff(slopes))
    _sum = np.sum(slopes)
    
    # If calculation results (either) too high = corruption -> fix
    if difference > slopeThreshold:
        issueIndex = np.argmax(slopes)
        if issueIndex == 0:
            points[1,1] = max(points[4,1], points[2,1])
        else:
            points[4,1] = max(points[1,1], points[2,1])
            
    if _sum > totalSlopeThreshold:
        points[1,1] = np.average(points[[4,2], 1])
        points[4,1] = np.average(points[[1,2], 1])
        points[2,1] = np.average(points[[4,1], 1])  
        
    return points

def align_face(image, eyePoints):
  # Function to rotate image to align the face
  # Get left eye & right eye coordinates
  leftEyeX,leftEyeY = eyePoints[0]
  rightEyeX, rightEyeY = eyePoints[1]
  
  # Calculate angle of rotation & origin point
  angle = math.atan( (leftEyeY - rightEyeY) / (leftEyeX - rightEyeX) ) * (180/math.pi)
  origin_point = tuple(np.array(image.shape[1::-1]) / 2)
  
  # Rotate using rotation matrix
  rot_mat = cv2.getRotationMatrix2D(origin_point, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def cropFullFace(image, points, padding = True, xProportion = 0.025, yProportion = 0.025):
    # Function to extract the face part of the image
    
    imageShape = image.shape
    # Get borders of the 4 directions
    top = points[:,1].min()
    bottom = points[:,1].max()
    left = points[:,0].min()
    right = points[:,0].max()
    
    if padding:
        # X-factor is a an additional proportion of the image on X-axis, considered in the output 
        # Y-factor is the same for Y-axis 
        xFactor = int((xProportion) * imageShape[1])
        yFactor = int((yProportion) * imageShape[0])
        x,y,x2,y2 = (max(left-xFactor, 0), max(top-yFactor, 0) ,min(right+yFactor, imageShape[0]), min(bottom+yFactor, imageShape[0]) )
    
    else:
        x,y,x2,y2 = (left,top ,right, bottom )

    cropped = image[y:y2, x:x2]
    return cropped


def delaunayOnPlane(facial_points):
    # Function to visualize delaunay triangulation on matplotlib
    tri = Delaunay(facial_points)
    rot = transforms.Affine2D().rotate_deg(180)
    base = plt.gca().transData
    plt.gca().invert_xaxis()
    plt.triplot(facial_points[:,0], facial_points[:,1], tri.simplices.copy(), transform=rot+base)
    plt.plot(facial_points[:,0], facial_points[:,1], 'o', transform=rot+base)
    plt.show()
    

def drawPoints(image, points, pointColor=(255,255,255), lineColor=(255,255,255), pointThickness=6, lineThickness=1):
    # Function to draw points on facial features
    for i in points:
        x,y = i
        image = cv2.circle(image, (x,y), radius=0, color=pointColor, thickness=pointThickness)

    return image


