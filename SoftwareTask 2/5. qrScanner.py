import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar

def input_img():
    while True:
        img_path = input("Enter the filename: ")
        img = cv2.imread(img_path)
        if img is not None:
            return img
        print("No such img file exists")

def barcode_scanner(img):
    decoded_objects = pyzbar.decode(img)

    for obj in decoded_objects:
        print("Type:", obj.type)
        print("Data:", obj.data.decode('utf-8'))

def main():
    img = input_img()
    barcode_scanner(img)

if __name__ == "__main__":
    main()
