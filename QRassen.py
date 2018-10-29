from __future__ import print_function
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2
import glob
import math

framesnamen=[]
framesnamen.append('LR.jpg')
framesnamen.append('links.jpg')
framesnamen.append('rechts.jpg')
framesnamen.append('50cm.jpg')
framesnamen.append('40cm.jpg')
framesnamen.append('50cm2.jpg')
framesnamen.append('60cm.jpg')
framesnamen.append('70cm.jpg')
framesnamen.append('75cm.jpg')
framesnamen.append('n50cm.jpg')
framesnamen.append('n60cm.jpg')


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
    vector3D[1]=(0,120,0)
    vector3D[2]=(120,120,0)
    vector3D[3]=(120,0,0)
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
    axis = np.float32([[120,0,0], [0,120,0], [0,0,-120]])
    imgpts, jac = cv2.projectPoints(axis,rvecs, tvecs, mtx, dist)
    frame = draw(frame,testarray,imgpts)
    cv2.imwrite('test%s.jpg'%welke,frame)
    #print(mtx)
    yh,xw=frame.shape[:2]
    #print(newcameramtx)
    fx=mtx[0][0]
    fy=mtx[1][1]
    cx=mtx[0][2]
    cy=mtx[1][2]
    focalLength=(fx+fy)/2
    print('       huidige afbeelding %s'%welke)
    print('focalLength %f' %focalLength)
    afstandY=math.sqrt(((punten2D[1][0]-punten2D[2][0])*1.0)**2+(1.0*(punten2D[1][1]-punten2D[2][1]))**2)
    #print('afstandY %f'%afstandY)
    #afstandcy=cx*10.0/afstandY
    #print('afstandcy %f' % afstandcy)
    afstandfy=focalLength*120.0/afstandY
    print('afstandfy %f mm' %afstandfy)
    afstandX=math.sqrt(((punten2D[0][0]-punten2D[1][0])*1.0)**2+(1.0*(punten2D[0][1]-punten2D[1][1]))**2)
    #print('afstandX %f'%afstandX)
    #afstandcx=cy*10.0/afstandX
    #print('afstandcx %f' %afstandcx)
    afstandfx=focalLength*120.0/afstandX
    print('afstandfx %f mm' %afstandfx)
    rotMat,_ = cv2.Rodrigues(rvecs)
    T0 = np.zeros((4, 4))
    T0[:3,:3] = rotMat
    T0[:4,3] = [0, 0, 0, 1]
    T0[:3,3] =  np.transpose(tvecs)
    cam=np.transpose([0, 0, 0, 1])
    transform=np.dot(T0,cam)
    maxval=np.amax(transform)
    print('positie camera:')
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    ax.plot([0 ,0],[0,(transform[1])],[0,0],color='blue')
    ax.plot([0 ,(transform[0])],[0,0],[0,0],color='green')
    ax.plot([0 ,0],[0,0],[0,(transform[2])],color='red')
    ax.plot([0 ,transform[0]],[0,transform[1]],[0,transform[2]],color='yellow')
    print(transform)
    plt.show()
    
    
   

    

  

for k in range(2,3):
  foto=cv2.imread(framesnamen[k])
  drawAxis(foto,framesnamen[k])
