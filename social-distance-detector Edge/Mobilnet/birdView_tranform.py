import numpy as np
import cv2

def rect_to_centroids(rect):
    """
    Compute all the centroid from rects
    IN -> rects bounding boxes
    OUT-> Centroid
    """
    centroid=[]
    for i in rect:
        if(i!=""):
            centroid.append(((i[2]+i[0])/2,(i[1]+i[3])/2))
        #centroid.append(((max(i[2], i[0])-min(i[2], i[0]))/2,(max(i[1], i[3]) - min(i[1], i[3]))/2))
    return centroid

def bird_transform(width,height,image,list_downoids):
    """ Compute the transformation matrix
    IN -> height, width : size of the image, list_downoids : list that contains the points to transform
    OUT -> return : list containing all the new points

    """
    # Create an array with the parameters (the dimensions) required to build the matrix
    img_params = np.float32([[0,0],[width,0],[0,height],[width,height]])
    # Compute and return the transformation matrix
    matrix = cv2.getPerspectiveTransform(img_params,img_params)
    # Compute the new coordinates of our points
    list_points_to_detect = np.float32(list_downoids).reshape(-1, 1, 2)
    
    #Trasnform the point using the matrix calculated befor
    transformed_points = cv2.perspectiveTransform(list_points_to_detect, matrix)
    
    # Loop over the points and add them to the list that will be returned
    transformed_points_list = list()
    for i in range(0,transformed_points.shape[0]):
        transformed_points_list.append([transformed_points[i][0][0],transformed_points[i][0][1]])
    return transformed_points_list
      
