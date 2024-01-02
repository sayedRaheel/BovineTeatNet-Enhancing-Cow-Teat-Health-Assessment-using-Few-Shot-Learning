
import cv2
def get_total_frame_number(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return None

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Release the video file and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

    return total_frames