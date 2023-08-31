import sys
import math
import cv2 as cv
import numpy as np


def main():
    default_file = 'sudoku.png'
    #filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread('bottom.png', cv.IMREAD_COLOR)
    srcg = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
    #print('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
    #return -1

    dst = cv.Canny(srcg, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    #cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(src)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv.LINE_AA)

    circles = cv.HoughCircles(srcg,cv.HOUGH_GRADIENT,1,20, param1=10,param2=40,minRadius=10,maxRadius=0)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv.circle(cdstP, (x, y), r, (0, 255, 0), 2)
            #cv.rectangle(cdstP, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    cv.imshow("Source", src)
    cv.imshow("gray", srcg)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    cv.waitKey()
    return 0


if __name__ == "__main__":
    main()