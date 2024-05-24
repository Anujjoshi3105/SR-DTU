import cv2
import numpy as np

def input_image():
    while True:
        img_path = input("Enter the filename: ")
        img = cv2.imread(img_path)
        if img is None:
            print("No such image file exists")
        else:
            width = int(input("Enter the width size(px): "))
            height = int(input("Enter the height size(px): "))
            return cv2.resize(img, (width, height))

def detect_and_draw_shapes(img, threshold_value=127):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue

        # Approximate the shape
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

        # Using drawContours() function
        cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)

        # Finding center point of shape
        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])

        # Putting shape name at the center of each shape
        if len(approx) == 3:
            shape_name = 'Triangle'
        elif len(approx) == 4:
            shape_name = 'Quadrilateral'
        elif len(approx) == 5:
            shape_name = 'Pentagon'
        elif len(approx) == 6:
            shape_name = 'Hexagon'
        else:
            shape_name = 'Circle'

        cv2.putText(img, shape_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Displaying the image after drawing contours
    cv2.imshow('shapes', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    img = input_image()
    threshold = int(input("Enter the threshold: "))
    detect_and_draw_shapes(img, threshold)

if __name__ == "__main__":
    main()
