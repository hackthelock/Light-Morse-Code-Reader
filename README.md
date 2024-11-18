# Light-Morse-Code-Reader
Python Script for Morse Code Detection via Light Blinks Using OpenCV

This script processes a video feed to detect light blinks representing Morse code, decode them, and provide a visualization. Here's how it works:

Video Processing:
The video is divided into individual frames, and each frame is analyzed to detect light intensity spikes.

Light Detection and Marking:
The regions of detected light spikes are marked in green, as shown in the first example image.

![image](https://github.com/user-attachments/assets/96f2c468-d734-43df-b6e2-667b6ba74b9c)


Duration Analysis:
The script calculates the number of consecutive frames with light (green) and dark (red), measuring durations:

Green (light durations): Represent dots (.) or dashes (-).
Red (dark durations): Indicate spaces between letters or words.
Threshold Calculation:
Thresholds for distinguishing dots, dashes, and spaces are calculated based on the durations:

Short light durations are interpreted as dots.
Longer light durations are interpreted as dashes.
Short dark durations signify spaces between characters.
Longer dark durations represent word boundaries.
Visualization:
A graphical display overlays the detected light (green) and dark (red) durations for easier interpretation.

![image](https://github.com/user-attachments/assets/c23ee89c-0875-4545-b990-a0880a641e50)

Decoding:
the script converts the detected sequences into text. While the decoding isn't perfect (sensitive to lighting conditions, blink duration, and brightness), it provides a functional baseline for translating light blinks into Morse code.

## Credits

This project is inspired by [MorseDecoder](https://github.com/new-silvermoon/MorseDecoder.git) by [new-silvermoon]. While it follows the same fundamental principles, it introduces enhanced features such as automatic threshold calculation. 


