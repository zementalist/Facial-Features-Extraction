from detection import *
from geometry import *
from extractor import *
import cv2
import os

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

def main():
    # Move opencv window
#    winname = "Test"
#    cv2.namedWindow(winname)
#    cv2.moveWindow(winname, 40,30) 
    
    # Capture all images in current folder & their names
    images, filesnames = load_images_from_folder('.')
    
    # Detect & Visualize each image
    for i in range(0,len(images)):
        originalImage = images[i]
#        cv2.imshow(winname, originalImage) 
#        cv2.waitKey(0)
        
        # Detect eyes landmarks, to align the face later
        eyePoints = facial_landmarks(originalImage, eyeOnlyMode=True)
        
        if eyePoints is not None:
            
            # Align face and redetect landmarks
            image = align_face(originalImage, eyePoints)
            improved_landmarks = facial_landmarks(image, allowEnhancement=True)

            # Extract feature
            options = ['all']
            feature = face_parts_imgs(originalImage, improved_landmarks, options)
            for key in feature:
                cv2.imwrite(key + '.jpg', feature[key])
                cv2.imshow(key, feature[key])
                cv2.waitKey(0)
            
            # Compare features, cluster & classify -> predict gender, personality, emotions.. whatever

    
            
            

main()