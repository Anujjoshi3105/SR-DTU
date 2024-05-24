import os
import cv2
import numpy as np

def input_image():
    while True:
        img_path = input("Enter the filename: ")
        img = cv2.imread(img_path)
        if img is None:
            print("No such image file exists")
            continue
        return cv2.resize(img, (640, 480))

def color_detection(image, lower_bound, upper_bound):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    return cv2.bitwise_and(image, image, mask=mask)

def create_slider_window():
    cv2.namedWindow("Color Detection Slider", cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow("Color Detection Slider", 640, 480)
    for trackbar in ["Hue Min", "Hue Max", "Saturation Min", "Saturation Max", "Value Min", "Value Max"]:
        cv2.createTrackbar(trackbar, "Color Detection Slider", 0, 255, lambda x: None)

def main():
    img = input_image()
    img = cv2.resize(img, (640, 480))
    create_slider_window()

    while True:
        lower_bound = np.array([cv2.getTrackbarPos(f"{color} Min", "Color Detection Slider") for color in ["Hue", "Saturation", "Value"]])
        upper_bound = np.array([cv2.getTrackbarPos(f"{color} Max", "Color Detection Slider") for color in ["Hue", "Saturation", "Value"]])

        resulting_img = color_detection(img, lower_bound, upper_bound)

        stacked_imgs = np.hstack([img, resulting_img])
        cv2.imshow("Color Detection", stacked_imgs)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            print(f"Final Values: Lower Bound={lower_bound}, Upper Bound={upper_bound}")
            cv2.imwrite("color_detected_image.jpg", resulting_img)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
