import cv2
import numpy as np

def add_subtitle(video_path, srt_path):
    cap = cv2.VideoCapture(video_path)

    with open(srt_path, 'r') as file:
        subtitles = file.read().split('\n\n')

    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = index * 33  # Assuming 30 fps, 1000ms / 30fps = 33ms per frame
        subtitle = get_subtitle_at_time(subtitles, current_time)

        if subtitle:
            text = subtitle['text']
            position = (50, 50)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 255, 255)
            thickness = 2

            cv2.putText(frame, text, position, font, font_scale, color, thickness)

        cv2.imshow('Video with Subtitle', frame)

        if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
            break

        index += 1

    cap.release()
    cv2.destroyAllWindows()

def get_subtitle_at_time(subtitles, time):
    for subtitle in subtitles:
        lines = subtitle.split('\n')
        time_range = lines[1].split(' --> ')
        start_time = convert_to_milliseconds(time_range[0])
        end_time = convert_to_milliseconds(time_range[1])

        if start_time <= time <= end_time:
            return {'start_time': start_time, 'end_time': end_time, 'text': '\n'.join(lines[2:])}

    return None

def convert_to_milliseconds(time_str):
    h, m, s = map(float, time_str.replace(',', '.').split(':'))
    return int((h * 3600 + m * 60 + s) * 1000)

if __name__ == "__main__:
    add_subtitle('1.mp4', '1.srt')
