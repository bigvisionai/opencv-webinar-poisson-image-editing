import cv2,argparse
import numpy as np

# Default radius
radius = 15

def findBestSquare(img, center):

  # Initializing the variables
  minVal = 1000.0
  nSquares = 3

  startPosition = np.array(center,dtype=int) - nSquares*radius
  newPosition = np.copy(startPosition)
  widthSquare = 2*radius

  #check all the squares in a nSquares x nSquares neighborhood around the required point
  for i in range(nSquares):
    for j in range(nSquares):

      #take a square from the neighborhood of the center
      newPosition[0] = startPosition[0] + i*widthSquare
      newPosition[1] = startPosition[1] + j*widthSquare

      #check to see if the square we want is out of the image if yes, proceed to next square
      if (newPosition[0] < 0 or newPosition[1] < 0 or
          newPosition[1] + widthSquare > img.shape[0] or
          newPosition[0] + widthSquare > img.shape[1]):
        continue

      checkSquare = img[newPosition[1]:newPosition[1] + widthSquare, newPosition[0]:newPosition[0] + widthSquare, :]

      #find a measure of roughness of the square block
      meanSobelX = np.mean(np.abs(cv2.Sobel(checkSquare, cv2.CV_32F, 1, 0 )))
      meanSobelY = np.mean(np.abs(cv2.Sobel(checkSquare, cv2.CV_32F, 0, 1 )))

      #if it is smoother than previous ones update the best square
      if (meanSobelX + meanSobelY) < minVal:
        minVal = meanSobelX + meanSobelY
        bestSquare = checkSquare
      else:
        continue

  return bestSquare

def onMouse(event, x, y, flags, param):
  global src, blemish, mask, center

  # If left click
  if event == cv2.EVENT_LBUTTONDOWN:

    center = (x, y)

    # check if the point lies on the boundary of the image
    if(x - radius < 0 or x + radius > src.shape[1] or y - radius < 0 or y + radius > src.shape[0]):
      return

    # catch the blemish region
    blemish = src[center[1] - radius : center[1] + radius, center[0] - radius : center[0] + radius, :]

    # find the smoothest region around the marked point
    smoothRegion = findBestSquare(src,center)

    # Create a white mask of the same size as the smooth region
    mask = np.zeros(smoothRegion.shape, smoothRegion.dtype)
    cv2.circle(mask, (radius, radius), radius, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    # Perform Seamless Cloning
    src = cv2.seamlessClone(smoothRegion, src, mask, center, cv2.NORMAL_CLONE)

    cv2.imshow("Blemish Remover",src)

  # added functionality for UNDO-ing the last modification
  elif event == cv2.EVENT_RBUTTONDOWN:

    # Revert back to the previous result
    src = cv2.seamlessClone(blemish, src, mask, center, cv2.NORMAL_CLONE)
    cv2.imshow("Blemish Remover",src)

if __name__ == '__main__' :

  ap = argparse.ArgumentParser()
  ap.add_argument("-f", "--filename", help="Path to the image")
  ap.add_argument("-r", "--radius",  help="radius of square of blemish")
  args = vars(ap.parse_args())

  filename = "blemish.png"

  # load the image and setup the mouse callback function
  global src

  if(args["filename"]):
    filename = args["filename"]

  src = cv2.imread(filename)

  if(args["radius"]):
    radius = int(args["radius"])
  print("Using a patch of radius {}".format(radius))

  cv2.namedWindow("Blemish Remover")
  cv2.setMouseCallback("Blemish Remover", onMouse)

  while True:

    # display the image and wait for a keypress
    cv2.imshow("Blemish Remover", src)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
      cv2.destroyAllWindows()
      break

    # Save the image if 's' is pressed on the keyboard
    if key == ord("s"):
      cv2.imwrite('clean_blemish.jpg',src)
