import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def input_image():
    while True:
        img_path = input("Enter the filename: ")
        img = cv2.imread(img_path)

        if img is None:
            print("No such image file exists")
            continue
        
        img_type = input("Enter the image type [c: color, g: gray, u: unchanged]: ").lower()
        if img_type in ['c', 'g', 'u']:
            return cv2.imread(img_path, 1 if img_type == 'c' else 0 if img_type == 'g' else -1)
        print("INCORRECT IMAGE TYPE. Please enter C, G, or U.")

def display_image(img, title):
    if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
        cv.imshow(title, img)
        print("Press 'esc key' to exit")
        if cv.waitKey(0) == 27:
            cv.destroyAllWindows()
    else:
        print("Error: Invalid image dimensions. Cannot display the image.")

def save_image(img, filename, extension):
    cv.imwrite(f"{filename}.{extension}", img)
    print(f"Image saved as {filename}")

def convert_color(img, color_code):
    conversion_code = cv.COLOR_BGR2GRAY if color_code == 'g' else cv.COLOR_BGR2HSV if color_code == 'h' else None
    try:
        return cv.cvtColor(img, conversion_code)
    except:
        print('INCORRECT COLOR CODE')
        return img

def split_channels(img):
    b, g, r = cv.split(img)
    zeros = np.zeros(img.shape[:2], dtype="uint8")
    display_image(cv.merge([zeros, zeros, r]), "Red")
    display_image(cv.merge([zeros, g, zeros]), "Green")
    display_image(cv.merge([b, zeros, zeros]), "Blue")

def remove_background(image):
    desired_width = int(input("Enter the width: "))
    initial_threshold = int(input("Enter the threshold: "))
    def on_trackbar_change(value):
        nonlocal blk_thresh
        blk_thresh = value
        print("Variable value:", blk_thresh)

    def value_scaling(value, min_value, max_value, new_min, new_max):
        scaled_value = (value - min_value) * (new_max - new_min) / (max_value - min_value) + new_min
        return int(scaled_value)

    # Resizing the image
    aspect_ratio = image.shape[1] / image.shape[0]
    desired_height = int(desired_width / aspect_ratio)
    resized_image = cv.resize(image, (desired_width, desired_height))

    # Initialize trackbar
    blk_thresh = initial_threshold
    scaled_thresh = value_scaling(blk_thresh, 0, 100, 0, 255)

    # Create a window and trackbar
    window_name = 'Background Removed'
    cv.namedWindow(window_name)
    cv.createTrackbar('Variable', window_name, scaled_thresh, 100, on_trackbar_change)

    while True:
        gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        _, threshold_img = cv.threshold(blur, blk_thresh, 255, cv.THRESH_BINARY)
        mask = 255 - threshold_img
        result = cv.bitwise_and(resized_image, resized_image, mask=mask)
        cv.imshow(window_name, result)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cv.destroyAllWindows()

    return result

def apply_blur(img, blur_type='gaussian', kernel_size=5):
    kernel_size = max(1, kernel_size) + (1 - max(1, kernel_size) % 2)

    blur_functions = {
        'g': cv.GaussianBlur,
        'm': cv.medianBlur,
        'a': cv.blur
    }

    blur_function = blur_functions.get(blur_type, cv.GaussianBlur)
    return blur_function(img, (kernel_size, kernel_size), 0)

def rotate_image(img, angle):
    rows, cols, _ = img.shape
    rotation_matrix = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    return cv.warpAffine(img, rotation_matrix, (cols, rows))

def scale_image(img, scale_factor):
    return cv.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_LINEAR)

def histogram_equalization(img):
    img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv.equalizeHist(img_yuv[:, :, 0])
    return cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)

def image_filter(img, filter_type='edge'):
    if filter_type == 'edge':
        input_edge1 = int(input("Enter the edge input1: "))
        input_edge2 = int(input("Enter the edge input2: "))
        return cv.Canny(img, input_edge1, input_edge2)
    elif filter_type == 'sharpen':
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        return cv.filter2D(img, -1, kernel)
    else:
        print("Invalid filter type. Using edge filter by default.")
        return cv.Canny(img, 100, 200)

def image_thresholding(img, threshold_value):
    _, thresholded_img = cv.threshold(img, threshold_value, 255, cv.THRESH_BINARY)
    return thresholded_img

def image_morphology(img, operation='erode', kernel_size=5):
    kernel_size = max(1, kernel_size) + (1 - max(1, kernel_size) % 2)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    iterations = int(input("Enter the number of iterations: "))

    morph_operations = {
        'e': cv.erode,
        'd': cv.dilate,
        'o': lambda img, kernel, iterations: cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=iterations),
        'c': lambda img, kernel, iterations: cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=iterations)
    }

    morph_operation_function = morph_operations.get(operation, cv.erode)
    return morph_operation_function(img, kernel, iterations=iterations)

def image_comparison(img1):
    img2 = input_image()
    img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))

    difference_img = cv.absdiff(img1, img2)
    title = input("Enter the title for the comparison result: ").capitalize()
    display_image(difference_img, title)

def image_cropping(img):
    x, y, w, h = cv.selectROI("Select ROI", img)
    roi = img[y:y+h, x:x+w]
    return roi

def image_flipping(img):
    flip_code = int(input("Enter flip code (0: horizontal, 1: vertical, -1: both): "))
    return cv.flip(img, flip_code)

def image_drawing(img):
    img_copy = img.copy()
    drawing = False
    mode = int(input("Enter the mode (0: circle, 1: rectangle): "))
    ix, iy = -1, -1

    def draw_shape(event, x, y, flags, param):
        nonlocal ix, iy, drawing, mode

        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv.EVENT_LBUTTONUP:
            drawing = False
            if mode == 1:
                cv.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv.circle(img_copy, (x, y), 5, (0, 0, 255), -1)

    cv.namedWindow('Image Drawing')
    cv.setMouseCallback('Image Drawing', draw_shape)

    while True:
        cv.imshow('Image Drawing', img_copy)
        if cv.waitKey(1) == 27:
            break

    cv.destroyAllWindows()
    return img_copy

def image_histogram(img):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.title('Image Histogram')
    plt.show()

def image_contours(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    low = int(input("Enter the lower limit: "))
    high = int(input("Enter the higher limit: "))
    _, thresh = cv.threshold(gray, low, high, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(img)

    cv.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    display_image(contour_img, "Contours")

def image_hough_circles(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    low = int(input("Enter the minimum radius: "))
    high = int(input("Enter the maximum radius: "))
    circles = cv.HoughCircles(
        gray, cv.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=low, maxRadius=high
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        circle_img = img.copy()

        for i in circles[0, :]:
            # Draw the outer circle
            cv.circle(circle_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv.circle(circle_img, (i[0], i[1]), 2, (0, 0, 255), 3)

        display_image(circle_img, "Hough Circles")
    else:
        print("No circles found.")

def image_canny_edge_detection(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    lower = int(input("Enter the lower limit: "))
    higher = int(input("Enter the higher limit: "))
    edges = cv.Canny(gray, lower, higher)
    display_image(edges, "Canny Edge Detection")

def main():
    img = input_image()
    while True:
        print("\n------Menu------")
        print("1. Input an image")
        print("2. Display the image")
        print("3. Save the image")
        print("4. Convert it to different color formats")
        print("5. Split image channels")
        print("6. Remove the image background")
        print("7. Apply image blurring")
        print("8. Rotate the image")
        print("9. Scale the image")
        print("10. Histogram Equalization")
        print("11. Apply Image Filter")
        print("12. Image Thresholding")
        print("13. Image Morphological Operations")
        print("14. Image Comparison")
        print("15. Image Cropping")
        print("16. Image Flipping")
        print("17. Image Drawing")
        print("18. Image Histogram")
        print("19. Image Contours")
        print("20. Hough Circles Detection")
        print("21. Canny Edge Detection")
        print("22. Exit the program")
        
        choice = input("Enter your choice (1-22): ")

        if choice == '1':
            img = input_image()
        elif choice == '2':
            title = input("Enter the title: ").capitalize()
            display_image(img, title)
        elif choice == '3':
            filename = input("Enter the filename to save: ")
            extension = input("Enter the extension [p: png/ j: jpg]: ").lower()
            save_image(img, filename, 'png' if extension == 'p' else 'jpg')
        elif choice == '4':
            color_option = input("Choose a color code option (g: BGR2GRAY, h: BGR2HSV): ").lower()
            img = convert_color(img, color_option)
            print("Image color converted.")
        elif choice == '5':
            split_channels(img)
            print("Image splitted.")
        elif choice == '6':
            img = remove_background(img)
        elif choice == '7':
            blur_type = input("Enter the blur type (g: gaussian, m: median, a: average): ").lower()
            kernel_size = int(input("Enter the odd kernel size (e.g., 5): "))
            img = apply_blur(img, blur_type, kernel_size)
            print("Image blurred. You can display/save the result.")
        elif choice == '8':
            angle = float(input("Enter the rotation angle in degrees: "))
            img = rotate_image(img, angle)
            print("Image rotated.")
        elif choice == '9':
            scale_factor = float(input("Enter the scaling factor: "))
            img = scale_image(img, scale_factor)
            print("Image scaled.")
        elif choice == '10':
            img = histogram_equalization(img)
            print("Histogram equalization applied.")
        elif choice == '11':
            filter_type = input("Enter the filter type (edge/sharpen): ").lower()
            img = image_filter(img, filter_type)
            print("Image filtered.")
        elif choice == '12':
            threshold_value = int(input("Enter the threshold value: "))
            img = image_thresholding(img, threshold_value)
            print("Image thresholded.")
        elif choice == '13':
            morph_operation = input("Enter the morphological operation (e: erode, d: dilate, o: open, c:close): ").lower()
            kernel_size = int(input("Enter the odd kernel size: "))
            img = image_morphology(img, morph_operation, kernel_size)
            print("Morphological operation applied.")
        elif choice == '14':
            image_comparison(img)
        elif choice == '15':
            img = image_cropping(img)
            print("Image cropped.")
        elif choice == '16':
            img = image_flipping(img)
            print("Image flipped.")
        elif choice == '17':
            img = image_drawing(img)
            print("Image drawing applied.")
        elif choice == '18':
            image_histogram(img)
        elif choice == '19':
            image_contours(img)
        elif choice == '20':
            image_hough_circles(img)
        elif choice == '21':
            image_canny_edge_detection(img)
        elif choice == '22':
            print("Exiting the program!!!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 22.")

if __name__ == "__main__":
    main()
