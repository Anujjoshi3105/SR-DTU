import cv2

def save_grayscale_video(input_filename, output_filename, fps):
    video = cv2.VideoCapture(input_filename)
    
    if not video.isOpened():
        print("Error opening the video file")
        return

    size = (int(video.get(3)), int(video.get(4)))
    result = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'MJPG'), 10, size, isColor=False)

    try:
        while True:
            ret, frame = video.read()

            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result.write(gray_frame)

            cv2.imshow('Gray Frame', gray_frame)

            if cv2.waitKey(fps) & 0xFF == ord('s'):
                break

    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        video.release()
        result.release()
        cv2.destroyAllWindows()

    print("Grayscale video successfully saved as", output_filename)

if __name__ == "__main__":
    input_filename = input("Enter the input video filename: ")
    output_filename = input("Enter the output video filename: ")
    fps = int(input("Enter the frame per milisecond: "))
    save_grayscale_video(input_filename, output_filename, fps)
