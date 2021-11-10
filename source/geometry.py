import math
import numpy as np

def equation1(points):
    # This method is a part of Eyebrows shape detector
    # Use shape_predictor_68 as a reference for points
    # points = [22,23,24,25,26]  (right eyebrow) - (method works for left eyebrow too)
    avgPoint = [np.average(points[[2,3],0]), np.average(points[[2,3],1])]
    angle = angle_of_3points(avgPoint,points[1],points[4])
    result = angle
    return result

def equation2(points):
    # This method is a part of Eyebrows shape detector
    # Use shape_predictor_68 as a reference for points
    # points = [22,23,24,25,26]  (right eyebrow) - (method works for left eyebrow too)
    result = slope(points[3], points[4])
    if result == "inf" or result == 0 :
        return 1
    return result

def equation3(points):
    # This method is a part of Eyebrows shape detector
    # Use shape_predictor_68 as a reference for points
    # points = [22,23,24,25,26]  (right eyebrow) - (method works for left eyebrow too)
    result = slope(points[1], points[2], True)
    if result == "inf" :
        result = 0
    x1,y1 = points[3]
    x2,y2 = points[4]
    slope2 = slope(points[3], points[4], True)
    if slope2 == "inf" :
        slope2 = 0
    result += slope2
    result = 1 if result == 0 else result
    return result


def equation4(points):
    # This method is a part of Eyebrows shape detector
    # Use shape_predictor_68 as a reference for points
    # points = [22,23,24,25,26]  (right eyebrow) - (method works for left eyebrow too)
    total = np.array([])
    for i in range(len(points)-2):
        total = np.append(total, (points[i+1,1] - points[i,1]))
        
    differences = abs(total[1]-total[0]) + abs(total[2] - total[1])
    
    slope0 = slope(points[0], points[1], True)
    slope1 = slope(points[2], points[3])
    slope2 = slope(points[3], points[4], True)
    
    slope0 = 1 if (slope0 == 0 or slope0 == "inf") else slope0    
    slope1 = 1 if (slope1 == 0 or slope1 == "inf") else slope1
    
    differences = 1 if differences == 0 else differences
    
    result = slope2 * (0.5*slope1/slope0) * (5/differences)
    #result = 1 if result >= 0.85 else 0.1
    result = 1 if result == 0 else result
    return result



def shape_area(points, circularArray=False):
    # Function to calculate area of any shape given its points coordinates
    # Circular array means that first point is added to the end of the array    
    result = 0
    for i in range(len(points)-1):
        x1,y1 = points[i]
        x2,y2 = points[i+1]
        result += (x1*y2) - (y1*x2)
    if not circularArray:
        x1,y1 = points[len(points)-1]
        x2,y2 = points[0]
        result += (x1*y2) - (y1*x2)
    result /= 2
    return abs(result)

        
def angle_of_3points(p1,p2,p3):
    # Function to get the angle of three points
    # NOTE : p1 is the middle point
    radian = math.atan2(p3[1] - p1[1], p3[0] - p1[0]) - math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    degrees = math.degrees(abs(radian))
    return degrees

def slope(point1, point2, absolute=False):
    x1,y1 = point1
    x2,y2 = point2
    deltaX = x2-x1
    deltaY = y2-y1
    if deltaX == 0:
        return "inf"
    slope = deltaY / deltaX
    if absolute:
        slope = abs(slope)
    return round(slope,3)


def diff_Yaxis(point1,point2):
    # Function to calculate the difference between 2 points on Y-axis
    return round(point1[1] - point2[1],3)

def eyeCenter(points):
    # Function to calculate the coordinates of the center of the eye
    p1,p2,p3,p4 = points
    x = np.average([p1[0],p2[0],p3[0],p4[0]])
    y = np.average([p1[1],p2[1],p3[1],p4[1]])
    return np.array([x,y])

def sum_difference(points):
    # Function to calculate the gradient difference in Y-axis for a group of points
    result = 0
    for i in range(points.shape[0]-1):
        result += diff_Yaxis(points[i], points[i+1])
    return round(result,3)


def sum_slopes(points, absolute=False):
    # Function to calculate the sum of slopes of a group of points
    result = 0
    for i in range(points.shape[0]-1):
        _slope = slope(points[i], points[i+1], absolute)
        if _slope == "inf":
            continue
        result += _slope
    return round(result,3)