import numpy as np
import pandas as pd
from geometry import *

def collectFaceComponents(facial_points):
    # Function to collect landmarks points, grouped, as polygons\shapes
    faceShape = np.concatenate((facial_points[0:17],facial_points[[78,74,79,73,80,71,70,69,76,75,77,0]])) 
    leftEye = np.concatenate((facial_points[36:42],np.array([facial_points[36]])))
    rightEye = np.concatenate((facial_points[42:47],np.array([facial_points[42]])))
    leftIBrow = facial_points[17:22]
    rightIBrow = facial_points[22:27]
    noseLine = facial_points[27:31]
    noseArc = facial_points[31:36]
    upperLip = facial_points[[50,51,52,63,62,61,50]]
    lowerLip = facial_points[[67,66,65,56,57,58,67]]
    faceComponents = {
            "face_shape":faceShape,
            "left_eye":leftEye,
            "right_eye":rightEye,
            "left_i_brow":leftIBrow,
            "right_i_brow":rightIBrow,
            "nose_line":noseLine,
            "nose_arc":noseArc,
            "upper_lip":upperLip,
            "lower_lip":lowerLip
            }
    return faceComponents

def face_parts_imgs(image, landmarks_points, options):
   # Facial feature extraction
   # Input:
   #    Image to process
   #    landmarks coordinates of the 81 landmark dlib
   #    options: array of strings, the features to be extracted
   #            mentioned below
   
   # Output:
   #    dictionary, {'feature_name': array(image)}
   
   
    # Get & Initialize face components
    faceComponents = collectFaceComponents(landmarks_points)
    face_shape = faceComponents["face_shape"]
    leftEye, rightEye = faceComponents["left_eye"], faceComponents["right_eye"]
    left_ibrow, right_ibrow = faceComponents["left_i_brow"], faceComponents["right_i_brow"]
    nose_line, nose_arc = faceComponents["nose_line"], faceComponents["nose_arc"]
    upper_lip = faceComponents["upper_lip"]
    
    if 'all' in options:
        options.extend(['forehead', 'left_eyebrow', 'right_eyebrow',
                        'both_eyebrow', 'clear_eyebrow', 'left_eye',
                        'right_eye', 'both_eye', 'clear_eye',
                        'left_eye_eyebrow', 'right_eye_eyebrow',
                        'both_eye_eyebrow', 'clear_eye_eyebrow',
                        'nose', 'mouth', 'eye_nose_mouth_eyebrow'
                        ])
    
    # Initialize response
    features = {}
    
    # Detect the clear face side (Better capture for eye+brows))
    # distance between nose bottom-point & eyes angle-point 
    lefteyeside = leftEye[3]
    righteyeside = rightEye[0]
    noseTip = nose_line[nose_line.shape[0]-1]
    if righteyeside[0] - noseTip[0] < 0: 
        # (in your perspective), person is looking to right direction -> so Left eye is clear
        clear_eye = leftEye
        clear_ibrow = left_ibrow
        clearer_left_side = True
    elif noseTip[0] - lefteyeside[0] < 0:
        # Person is looking to left direction -> so right eye is clear
        clear_eye = rightEye
        clear_ibrow = right_ibrow
        clearer_left_side = False
    else:
        # Decide which side is clearer (person is slightly looking to right or left)
        nose_eye_diff = abs(noseTip[0]-lefteyeside[0]) - abs(noseTip[0]-righteyeside[0])
        ibrow_position = "right" if nose_eye_diff <= 1 else "left"
        clear_eye = faceComponents[ibrow_position+"_eye"]
        clear_ibrow = faceComponents[ibrow_position+"_i_brow"]
        clearer_left_side = True if ibrow_position == 'left' else False
        
    ##### Forehead #####
    if 'forehead' in options:
        x, y, x2, y2 = np.min(face_shape[:, 0]), np.min(face_shape[:, 1]), np.max(face_shape[:, 0]), np.min(clear_ibrow[:, 1])
        forehead_img = image[y:y2, x:x2] # Best resolution (224, 64)
        features['forehead'] = forehead_img
        
    
    ##### Left eyebrow #####
    if 'left_eyebrow' in options:
        # x = between left eyebrow and left side faceshape landmark [index 0 of 81]
        # x2 = nose top landmark (between eyebrows)
        # y = top eyebrow landmark,   y2 = bottom eyebrow landmark
        x = int((left_ibrow[0, 0] + face_shape[0, 0]) / 2)
        x2 =  nose_line[0, 0]
        y, y2 = np.min(left_ibrow[:, 1]), np.max(left_ibrow[:, 1])
        left_ibrow_img = image[y:y2, x:x2]
        features['left_eyebrow'] = left_ibrow_img
    
    ##### Right eyebrow #####
    if 'right_eyebrow' in options:
        # x =  nose top landmark (between eyebrows)
        # x2 = between right eyebrow and right side faceshape landmark [index 16 of 81]
        # y = top eyebrow landmark,   y2 = bottom eyebrow landmark
        # y2 = eyebrow bottom landmark
        x = nose_line[0,0]
        x2 = int((right_ibrow[4, 0] + face_shape[16, 0]) / 2)
        y, y2 = np.min(right_ibrow[:, 1]), np.max(right_ibrow[:, 1])  
        right_ibrow_img = image[y:y2, x:x2]
        features['right_eyebrow'] = right_ibrow_img
    
    
    ##### Left eye #####
    if 'left_eye' in options:
        # x = between left eye and left side faceshape landmark [index 0 of 81]
        # x2 = top landmark of nose (between eyes)
        # y = between eye top landmark & eyebrow top landmark
        # y2 = second top nose landmark
        x = int((leftEye[0, 0] + face_shape[0, 0]) / 2)
        x2 = nose_line[0, 0]
        y = int((np.min(left_ibrow[:, 1]) + np.min(leftEye[:, 1])) / 2)
        y2 = nose_line[1, 1]
        leftEye_img = image[y:y2, x:x2]
        features['left_eye'] = leftEye_img
    
    ##### Right eye #####
    if 'right_eye' in options:
        # x = top landmark of nose (between eyes)
        # x2 = between right eye and right side faceshape landmark [index 16 of 81]
        # y = between eye top landmark & eyebrow top landmark
        # y2 = second top nose landmark
        x = nose_line[0, 0]
        x2 = int((rightEye[4, 0] + face_shape[16, 0]) / 2)
        y = int((np.min(right_ibrow[:, 1]) + np.min(rightEye[:, 1])) / 2)
        y2 = nose_line[1, 1]
        rightEye_img = image[y:y2, x:x2]
        features['right_eye'] = rightEye_img
    
    ##### Both eyebrows #####
    if 'both_eyebrow' in options:
        # x = between left eyebrow and left side faceshape landmark [index 0 of 81]
        # x2 = between right eyebrow and right side faceshape landmark [index 16 of 81]
        # y = top landmark of left/right eyebrow (maximum top is selected)
        # y2 = bottom landmark of left/right eyebrow (maximum bottom is selected)
        x = int((left_ibrow[0, 0] + face_shape[0, 0]) / 2)
        x2 = int((right_ibrow[4, 0] + face_shape[16, 0]) / 2)
        y = min(np.min(left_ibrow[:, 1]), np.min(right_ibrow[:, 1]))
        y2 = max(np.max(left_ibrow[:, 1]), np.max(right_ibrow[:, 1]))
        both_eyebrows_img = image[y:y2, x:x2]
        features['both_eyebrow'] = both_eyebrows_img
    
    ##### Both eyes #####
    if 'both_eye' in options:
        # x = between left eye and left side faceshape landmark [index 0 of 81]
        # x2 = between right eye and right side faceshape landmark [index 16 of 81]
        # y = between clear eyebrow & clear eye
        # y2 = second top nose landmark
        x = int((leftEye[0, 0] + face_shape[0, 0]) / 2)
        x2 = int((rightEye[4, 0] + face_shape[16, 0]) / 2)
        y = int((np.min(clear_ibrow[:, 1]) + np.min(clear_eye[:, 1])) / 2)
        y2 = nose_line[1, 1]
        both_eyes_img = image[y:y2, x:x2]
        features['both_eye'] = both_eyes_img
    
    ##### Eye and Eyebrow LEFT #####
    if 'left_eye_eyebrow' in options:
        # x = between left eyebrow and left side faceshape landmark [index 0 of 81]
        # x2 = nose top landmark (between eyebrows)
        # y = top left eyebrow landmark
        # y2 = second top nose landmark
        x = int((left_ibrow[0, 0] + face_shape[0, 0]) / 2)
        x2 =  nose_line[0, 0]
        y = np.min(left_ibrow[:, 1])
        y2 = nose_line[1, 1]
        eye_eyebrow_left =  image[y:y2, x:x2]
        features['left_eye_eyebrow'] = eye_eyebrow_left
    
    ##### Eye and Eyebrow RIGHT #####
    if 'right_eye_eyebrow' in options:
        # x = top landmark of nose (between eyes)
        # x2 = between right eyebrow and right side faceshape landmark [index 16 of 81]
        # y = top right eyebrow landmark
        # y2 = second top nose landmark
        x = nose_line[0, 0]
        x2 = int((right_ibrow[4, 0] + face_shape[16, 0]) / 2)
        y = np.min(right_ibrow[:, 1])
        y2 = nose_line[1, 1]
        eye_eyebrow_right = image[y:y2, x:x2]
        features['right_eye_eyebrow'] = eye_eyebrow_right
    
    ##### Eye and Eyebrow LEFT & RIGHT #####
    if 'both_eye_eyebrow' in options:
        # x = between left eyebrow and left side faceshape landmark [index 0 of 81]
        # x2 = between right eyebrow and right side faceshape landmark [index 16 of 81]
        # y = top eyebrow landmark
        # y2 = second top nose landmark
        x = int((left_ibrow[0, 0] + face_shape[0, 0]) / 2)
        x2 = int((right_ibrow[4, 0] + face_shape[16, 0]) / 2)
        y = min(np.min(left_ibrow[:, 1]), np.min(right_ibrow[:, 1]))
        y2 = nose_line[1, 1]
        eye_eyebrow_all = image[y:y2, x:x2]
        features['both_eye_eyebrow'] = eye_eyebrow_all
    
    
    ##### Clear Eyebrow #####
    if 'clear_eyebrow' in options:
        # x =  left face side OR nose top landmark (between eyebrows)
        # x2 = between clearer eyebrow and clearer side faceshape landmark [index 16 of 81]
        # y = top eyebrow landmark
        # y2 = eyebrow bottom landmark
        x = int((left_ibrow[0, 0] + face_shape[0, 0]) / 2) if clearer_left_side else nose_line[0,0]
        x2 = nose_line[0,0] if clearer_left_side else int((right_ibrow[4, 0] + face_shape[16, 0]) / 2)
        y, y2 = np.min(clear_ibrow[:, 1]), np.max(clear_ibrow[:, 1])  
        clear_ibrow_img = image[y:y2, x:x2]
        features['clear_eyebrow'] = clear_ibrow_img
    
    
    ##### Clear Eye #####
    if 'clear_eye' in options:
        # x = leftEye.x OR rightEye.x
        # x2 = leftEye.x2 OR rightEye.x2
        # y = between clear eyebrow and clear eye
        # y2 = second top nose landmark
        x = int((leftEye[0, 0] + face_shape[0, 0]) / 2) if clearer_left_side else nose_line[0, 0]
        x2 = nose_line[0, 0] if clearer_left_side else int((rightEye[4, 0] + face_shape[16, 0]) / 2)
        y = int((np.min(clear_ibrow[:, 1]) + np.min(clear_eye[:, 1])) / 2)
        y2 = nose_line[1,1]
        clear_eye_img = image[y:y2, x:x2]
        features['clear_eye'] = clear_eye_img
        
    
    ##### Clear Eye and Eyebrow #####
    if 'clear_eye_eyebrow' in options:
        # x =  left face side OR nose top landmark (between eyebrows)
        # x2 = between clearer eyebrow and clearer side faceshape landmark [index 16 of 81]
        # y = top eyebrow landmark
        # y2 = second top nose landmark
        x = int((left_ibrow[0, 0] + face_shape[0, 0]) / 2) if clearer_left_side else nose_line[0,0]
        x2 = nose_line[0,0] if clearer_left_side else int((right_ibrow[4, 0] + face_shape[16, 0]) / 2)
        y = np.min(clear_ibrow[:, 1])
        y2 = nose_line[1,1]
        clear_eye_eyebrow = image[y:y2, x:x2]
        features['clear_eye_eyebrow'] = clear_eye_eyebrow
        
    
    ##### Nose #####
    if 'nose' in options:
        # x = the most right landmark of left eye || nose bottom landmark if it's more left
        # x2 = the most left landmark of right eye|| nose bottom landmark if it's more right
        # y = average point on Y-axis of eyebrow
        # y2 = upper lip top landmark
        x = min(leftEye[3,0], nose_arc[0, 0])
        x2 = max(rightEye[0,0], nose_arc[4, 0])
        y = int(np.average(clear_ibrow[:, 1]))
        y2 = upper_lip[2, 1]
        nose = image[y:y2, x:x2]
        features['nose'] = nose
        

    ##### Mouth #####
    if 'mouth' in options:
        # x = left cheek [index 5 of 81]
        # x2 = right cheek [index 11 of 81]
        # y = nose bottom landmark
        # y2 = point between chin bottom landmark and lower lip [index 68 of 81]
        x = face_shape[5,0]
        x2 = face_shape[11,0]
        y = nose_arc[2, 1]
        y2 = landmarks_points[8,1] - int((landmarks_points[8,1]-landmarks_points[57,1]) / 2)
        mouth = image[y:y2, x:x2]
        features['mouth'] = mouth
    
    ##### Eyebrow Eye Nose Mouth ##### 
    if 'eye_nose_mouth_eyebrow' in options:
        # x = between left eyebrow and left side faceshape landmark [index 0 of 81]
        # x2 = between right eyebrow and right side faceshape landmark [index 16 of 81]
        # y = top eyebrow landmark
        # y2 =point between chin bottom landmark and lower lip [index 68 of 81]
        x = int((left_ibrow[0, 0] + face_shape[0, 0]) / 2)
        x2 = int((right_ibrow[4, 0] + face_shape[16, 0]) / 2)
        y = min(np.min(left_ibrow[:, 1]), np.min(right_ibrow[:, 1]))
        y2 = landmarks_points[8,1] - int((landmarks_points[8,1]-landmarks_points[57,1]) / 2)
        general_features = image[y:y2, x:x2]
        features['eye_nose_mouth_eyebrow'] = general_features
    
    return features

    