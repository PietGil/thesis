from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2
import glob
import math

frames=[]
frames.append(cv2.imread('LR.jpg'))
frames.append(cv2.imread('links.jpg'))
frames.append(cv2.imread('rechts.jpg'))
frames.append(cv2.imread('50cm.jpg'))


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


def drawAxis(frame,welke):
  decodedObjects=pyzbar.decode(frame)
  for decodedObject in decodedObjects: 
      points = decodedObject.polygon
      
      
      if len(points) > 4 : 
        punten2D = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
        punten2D= list(map(tuple, np.squeeze(punten2D)))
      else : 
        punten2D = points
      ##print(punten2D)
      #for i in range(0,4):
          #cv2.putText(frame,'(%d,%d)'% (punten2D[i][0], punten2D[i][1]),punten2D[i],cv2.FONT_HERSHEY_DUPLEX,2,(0,255,0),2,cv2.LINE_AA)

  if(len(decodedObjects)>0):
    vector3D=np.zeros((4,3), np.float32)
    vector3D[0]=(0,0,0)
    vector3D[1]=(0,10,0)
    vector3D[2]=(10,10,0)
    vector3D[3]=(10,0,0)
    testarray=np.zeros((4,1,2), np.float32)

    for i in range(0,4):
      testarray[i][0][0]=np.float32(punten2D[i][0])
      testarray[i][0][1]=np.float32(punten2D[i][1])
    points3D=[]
    lijst2D=[]
    lijst2D.append(testarray)
    points3D.append(vector3D)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ##print(testarray.shape)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(points3D, lijst2D,gray.shape[::-1],None,None)
    h,  w = frame.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    # undistort
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult.png',dst)

    #print(ret)
    #print (mtx)
    #print (dist)
    #print (rvecs)
    #print( tvecs)
    _,rvecs, tvecs, inliers = cv2.solvePnPRansac(vector3D, testarray, mtx, dist)
    axis = np.float32([[10,0,0], [0,10,0], [0,0,-10]])
    imgpts, jac = cv2.projectPoints(axis,rvecs, tvecs, mtx, dist)
    frame = draw(frame,testarray,imgpts)
    cv2.imwrite('test%d.jpg'%welke,frame)
    print(mtx)
    yh,xw=frame.shape[:2]
    #print(newcameramtx)
    fx=mtx[0][0]
    fy=mtx[1][1]
    cx=mtx[0][2]
    cy=mtx[1][2]
    focalLength=(fx+fy)/2
    print('focalLength %f' %focalLength)
    afstandY=math.sqrt(((punten2D[1][0]-punten2D[2][0])*1.0)**2+(1.0*(punten2D[1][1]-punten2D[2][1]))**2)
    print('volgende %d'%welke)
    #print('afstandY %f'%afstandY)
    #afstandcy=cx*10.0/afstandY
    #print('afstandcy %f' % afstandcy)
    afstandfy=focalLength*10.0/afstandY
    print('afstandfy %f' %afstandfy)
    afstandX=math.sqrt(((punten2D[0][0]-punten2D[1][0])*1.0)**2+(1.0*(punten2D[0][1]-punten2D[1][1]))**2)
    #print('afstandX %f'%afstandX)
    #afstandcx=cy*10.0/afstandX
    #print('afstandcx %f' %afstandcx)
    afstandfx=focalLength*10.0/afstandX
    print('afstandfx %f' %afstandfx)
    fafstand=math.sqrt((afstandfx)**2+(afstandfy)**2)
    print('fafstand %f' %fafstand)
    #cafstand=math.sqrt((afstandcx)**2+(afstandcy)**2)
    #print('cafstandfx %f' %cafstand)

    

  

for k in range(0,len(frames)):
  drawAxis(frames[k],k)
