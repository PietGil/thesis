from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2

def decode(im) : 
  # Find barcodes and QR codes
  decodedObjects = pyzbar.decode(im)#decodes datamatrix barcode in image

  # Print results
  for obj in decodedObjects:
    print('Type : ', obj.type)#type uit printen
    print('Data : ', obj.data,'\n')#wat de qr code inhoud uitprinten
    
  return decodedObjects


# Display barcode and QR code location  
def display(im, decodedObjects):

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
  cv2.imshow("Results", im)
  cv2.waitKey(0)

  
cap =cv2.VideoCapture('testvideo.mp4')
length=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(length)
succes,frame =cap.read()

while(succes):  
  cv2.imshow('frame',frame)
  if cv2.waitKey(1)& 0xFF==ord('q'):#sluiten als de video afgespeeld is
    break
  succes,frame =cap.read()

cap.release()
cv2.destroyAllWindows()




# # Main 
# if __name__ == '__main__':

#   # Read image
#   im = cv2.imread('foto.jpg')

#   decodedObjects = decode(im) #lijst van gedecodeerde waarden uit de afbeelding
# display(im, decodedObjects)
