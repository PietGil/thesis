from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2


# Display barcode and QR code location  
def imagepro(im, decodedObjects):

  # Loop over all decoded objects
  for decodedObject in decodedObjects: 
    points = decodedObject.polygon

    # If the points do not form a quad, find convex hull
    if len(points) > 4 : 
      hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))#convexhull veelhoek rond de punten
      hull = list(map(tuple, np.squeeze(hull)))#tuple zorgt ervoor dat de waarden niet veranderd kunnen worden #squeeze verwijdert de rijen waarvan de lengte 1 is
    else : 
      hull = points
    
    # Number of points in the convex hull
    n = len(hull)

    # Draw the convext hull
    for j in range(0,n):
      cv2.line(im, hull[j], hull[ (j+1) % n], (255,0,0), 3)#% teken is om terug nr de eerste lijn te gaan laatste cijfers zijn kleur en dikte

  # Display results 
  return im
  
def storeResult(im, decodedObjects):

  # Loop over all decoded objects
  for decodedObject in decodedObjects: 
    points = decodedObject.polygon

    # If the points do not form a quad, find convex hull
    if len(points) > 4 : 
      hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))#convexhull veelhoek rond de punten
      hull = list(map(tuple, np.squeeze(hull)))#tuple zorgt ervoor dat de waarden niet veranderd kunnen worden #squeeze verwijdert de rijen waarvan de lengte 1 is
    else : 
      hull = points
    
    # Number of points in the convex hull
    n = len(hull)

    # Draw the convext hull
    for j in range(0,n):
      cv2.line(im, hull[j], hull[ (j+1) % n], (255,0,0), 3)#% teken is om terug nr de eerste lijn te gaan laatste cijfers zijn kleur en dikte

  # Store results 
    cv2.imwrite("frame%d.jpg"%count,im)
    
  
count=0
cap =cv2.VideoCapture('4qr.mp4')
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out =cv2.VideoWriter('output.avi',fourcc,30,(1920,1080))
succes,frame =cap.read()
listofstrings=["list of barcodes:"]
currentstrings=[""]
vorig="nothing detected"
count=0
font= cv2.FONT_HERSHEY_DUPLEX
while(succes):    
  decodedObjects=pyzbar.decode(frame)
  if(len(decodedObjects)>0):
    frame=imagepro(frame,decodedObjects)
    #methode kopieren? tekst boven Qr code plaatsen
    for obj in decodedObjects:
      currentstrings.append('current -> Type : %s ->Data : %s'  %(obj.type,obj.data))
      if ('Data : '+obj.data) not in listofstrings:
        listofstrings.append('Next QR-code in frame %d'%count)
        listofstrings.append('Data : '+obj.data)
        listofstrings.append('Type : '+obj.type)
        print("frame%d.jpg"%count)
        print('Type : ', obj.type)
        print('Data : ', obj.data,'\n')
  plek=0
  for str in listofstrings:
      cv2.putText(frame,str,(100,100+plek),font,1,(0,255,0),1,cv2.LINE_AA)
      plek+=25
  for str in currentstrings:
      cv2.putText(frame,str,(100,100+plek),font,1,(0,255,0),1,cv2.LINE_AA)
      plek+=25
  out.write(frame)
  cv2.imshow('frame',frame)
  currentstrings=[""]

  if cv2.waitKey(1)& 0xFF==ord('q'):#sluiten als de video afgespeeld is
    break
  succes,frame =cap.read()
  count+=1

cap.release()
cv2.destroyAllWindows()