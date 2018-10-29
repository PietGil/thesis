from __future__ import print_function
import numpy as np
import cv2
import glob
import math

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('schaak50L.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        

cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print(mtx)
def draw(img, corners, imgpts,flen):
    corner = tuple(corners[0].ravel())
    b=corner
    a=tuple(imgpts[0].ravel())
    afstandX=math.sqrt(((a[0]-b[0])*1.0)**2+(1.0*(a[1]-b[1]))**2)
    afstandfy=flen*28.0/afstandX
    print(afstandfy)

    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]])


for fname in glob.glob('schaak50L.jpg'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        _,rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
        # rvecs=a[0]
        # tvecs=a[1]
        # inliers=a[2]
        # print(rvecs)
        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis,rvecs, tvecs, mtx, dist)

        img = draw(img,corners2,imgpts,mtx[0][0])
        cv2.imwrite('schaak.png', img)

cv2.destroyAllWindows()
