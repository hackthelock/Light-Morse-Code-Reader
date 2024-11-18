import cv2
import numpy as np
from tkinter import Tk
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from sklearn.cluster import KMeans
from pyMorseTranslator import translator
import time
from Xlib import display
from pynput.keyboard import Key, Controller
# morse dictionary
MORSE_CODE_DICT = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F',
    '--.': 'G', '....': 'H', '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L',
    '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P', '--.-': 'Q', '.-.': 'R',
    '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X',
    '-.--': 'Y', '--..': 'Z', '-----': '0', '.----': '1', '..---': '2',
    '...--': '3', '....-': '4', '.....': '5', '-....': '6', '--...': '7',
    '---..': '8', '----.': '9'
}

encoder=translator.Encoder()


def is_caps_lock_on():

    DISPLAY = display.Display()
    keyb = DISPLAY.get_keyboard_control()
    led = keyb._data['led_mask']
    return (led & 1) == 1

def toggle_caps_lock(keyboard):
    keyboard.press(Key.caps_lock)
    keyboard.release(Key.caps_lock)

def blink_caps_lock(delay_array):

    keyboard = Controller()
    for blink in (delay_array):
        toggle_caps_lock(keyboard)
        time.sleep(blink)


def morse_code_to_blinks(code,duration):
    duration=duration
    delay_array=[]
    for i in code:
        if i == '.':
            delay_array.append(duration)

        elif i == '-':
            delay_array.append(duration * 3)

        elif i == ' ':
            delay_array.append(duration * 7)


    return delay_array

def process_frame(frame, light_times, dark_times, light_times_count, dark_times_count):

    # Convert frame to grayscale and blur to reduce noise
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (11, 11), 0)

    # Convert to binary image
    _, binary_frame = cv2.threshold(blurred_frame, 200, 255, cv2.THRESH_BINARY)

    # Enhance image using erosion and dilation
    binary_frame = cv2.erode(binary_frame, None, iterations=2)
    binary_frame = cv2.dilate(binary_frame, None, iterations=4)

    # Find coordinates of light pixels
    light_coords = np.argwhere(binary_frame == 255)

    if len(light_coords) > 0:
        # Process light period
        if dark_times_count > 0:
            dark_times.append(dark_times_count)
            dark_times_count = 0

        light_times_count += 1

        # Visualize light source midpoint (Optional)
        mid_y, mid_x = np.mean(light_coords, axis=0).astype(int)
        cv2.circle(frame, (mid_x, mid_y), 5, (0, 255, 0), -1)

    else:
        # Process dark period
        if light_times_count > 0:
            light_times.append(light_times_count)
            light_times_count = 0

        dark_times_count += 1

    return light_times_count, dark_times_count


def read_from_file():

    # Open file dialog to select video file
    Tk().withdraw()
    video_path = askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4;*.avi;*.mkv;*.mov")]
    )

    if not video_path:
        print("No file selected. Exiting...")
        return [], []

    # Initialize variables
    light_times, dark_times = [], []
    light_times_count = dark_times_count = 0

    # Open video file
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Unable to open video file '{video_path}'.")
        return [], []

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {int(fps)}")
    frame_delay = int(1000 / fps)  # Calculate frame delay in milliseconds

    while True:
        # Read video frame
        read_flag, frame = video_capture.read()
        if not read_flag:
            break

        # Process the current frame
        light_times_count, dark_times_count = process_frame(
            frame, light_times, dark_times, light_times_count, dark_times_count
        )

        # Display the frame
        cv2.imshow("Processed Video", frame)

        # Synchronize with the video FPS
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()

    print("Light Durations:", light_times)
    print("Dark Durations:", dark_times[1:] if dark_times else [])
    return light_times, dark_times[1:] if dark_times else []



def find_thresholds(durations, n_clusters):
    """
    Finds thresholds for durations using K-means clustering.
    """
    durations = np.array(durations).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(durations)
    thresholds = sorted(kmeans.cluster_centers_.flatten())
    return thresholds


def analyze_durations(light_times, dark_times, error_margin=5):

    # Find the thresholds using K-means clustering
    light_thresholds = find_thresholds(light_times, n_clusters=2)
    dark_thresholds = find_thresholds(dark_times, n_clusters=3)

    # Apply error margin directly within this function
    light_thresholds = [threshold + error_margin for threshold in light_thresholds]
    dark_thresholds = [threshold + error_margin for threshold in dark_thresholds]

    return light_thresholds, dark_thresholds


def map_durations_to_morse(light_times, dark_times, light_thresholds, dark_thresholds):

    symbols = []

    for i, light_duration in enumerate(light_times):
        # Map light duration to dot or dash
        if light_duration < light_thresholds[0]:
            symbols.append('.')  # Dot

        else:
            symbols.append('-')  # Dash


        # Map the corresponding dark duration to spaces, if it exists
        if i < len(dark_times):
            dark_duration = dark_times[i]
            if dark_duration <= dark_thresholds[0]:
                continue  # Ignore short gaps
            elif dark_duration > dark_thresholds[0] and dark_duration<dark_thresholds[2] :
                symbols.append(' ')  # Space between letters

            else:
                # Append word space only if it's not at the end of the sequence
                if i < len(light_times) - 1:
                    symbols.append('   ')  # Space between words


    return ''.join(symbols).strip()


def morse_to_text(morse_string):

    words = morse_string.split('   ')  # Split Morse code into words
    decoded_message = []

    for word in words:
        letters = word.split(' ')  # Split each word into letters
        decoded_word = ''.join(MORSE_CODE_DICT.get(letter, '') for letter in letters)
        decoded_message.append(decoded_word)

    return ' '.join(decoded_message)



def visualize_durations_with_thresholds(light_times, dark_times, light_thresholds, dark_thresholds):

    # Combine light and dark times into a single sequence
    durations = []
    labels = []

    for i in range(max(len(light_times), len(dark_times))):
        if i < len(light_times):
            durations.append(light_times[i])
            labels.append(f"Light {i+1}")
        if i < len(dark_times):
            durations.append(dark_times[i])
            labels.append(f"Dark {i+1}")

    # Create a bar graph
    plt.figure(figsize=(12, 6))
    bar_colors = ['green' if 'Light' in label else 'red' for label in labels]
    plt.bar(range(len(durations)), durations, color=bar_colors)
    plt.xticks(range(len(durations)), labels, rotation=45, ha="right")
    plt.xlabel("Flash Sequence")
    plt.ylabel("Duration (Frames)")
    plt.title("Light and Dark Durations with Thresholds")

    # Add thresholds as horizontal lines
    for threshold in light_thresholds:
        plt.axhline(y=threshold, color='green', linestyle='--', label=f'Light Threshold: {threshold}')
    for threshold in dark_thresholds:
        plt.axhline(y=threshold, color='red', linestyle='--', label=f'Dark Threshold: {threshold}')

    # Add legend
    plt.legend(loc="upper right")
    plt.tight_layout()

    # Show the graph
    plt.show()



light_times, dark_times = read_from_file()
print("can read file ")

if not light_times or not dark_times:
    print("No data to process.")
else:


    light_thresholds, dark_thresholds = analyze_durations(light_times, dark_times)
    print(type(dark_thresholds[0]))
    print(f"Light Thresholds: {light_thresholds}")
    print(f"Dark Thresholds: {dark_thresholds}")
    visualize_durations_with_thresholds(light_times, dark_times, light_thresholds, dark_thresholds)

    # Map durations to binary code
    binary_code = map_durations_to_morse(light_times, dark_times, light_thresholds, dark_thresholds)
    print(f"Morse Code: {binary_code}")

    # Decode binary code to text
    #decoded_message = morse_to_text(binary_code)
    #print(f"Decoded Message: {decoded_message}")



