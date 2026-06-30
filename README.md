# Virtual_Violin
A virtual violin that you can play on, achieved by using Ai model to detect your fingure and arm movement. 

### OVERVIEW & FEATURES & MOTIVATION

I was amazed by the elegance of Kafka playing an invisible violin while I was playing the game Honkai: Star Rail. I don't know if you get it, but it just looks absolutely stunning.

I wanted to recreate it, but how do you even create something that physically doesn't exist yet can be played like a real violin? While working on my previous AI body posture monitoring project, it hit me, why not use AI vision to track finger and arm movements, and generate corresponding sounds based on those gestures? This program is my attempt to actually build something close to the fiction. It was incredibly challenging and time consuming, at least for me.
The current lightweight AI detection is not very accurate, and I was unable to make it identify the micro-movements of fingers required to play a real physical violin. (I mean, a more powerful AI model and a more sophisticated algorithm could definitely achieve that). Hence, I had to modify the required finger postures so the AI model could identify them reliably.
So, instead of the subtle string-pressing gestures used in real violin playing, the pitch is now controlled by how many fingers are raised and their specific binary arrangements.

### HOW IT WORKS 

So this program uses Google's MediaPipe Hand Landmarker to track both hands's posture.
It uses OpenCV , MediaPipe , Pygame&Numpy .

Left Hand (To control pitch):
The program reads the binary state (bent or straight) of your 4 fingers like 1101 0101 etc.
These combinations are mapped to specific musical notes (from `do` to `high mi`).
(I initially mapped all 16 possible binary combinations, but realized some gestures (like     bending only the ring finger) are physically anti-human. So, I redesigned the `GESTURE_MAP` to only include comfortable finger combinations, assigning the awkward ones to less used notes )
-Sticking your thumb out acts as a "Rest" signal, instantly muting the audio.

Right Hand (The bow):
The program tracks the movement distance of your right hand frame by frame.
If the right hand is moving (distance > threshold), the audio plays. If it stops, the audio fades out smoothly.

Audio:
Instead of loading audio files, the program uses `numpy` to generate sine waves mathematically based on the exact frequencies of musical notes (e.g., A4 = 440Hz). 
It then uses `pygame.sndarray` to convert these mathematical arrays into playable sound in real-time.

### INSTALLATION
1: Install everything , put the following into terminal
   `pip install opencv-python mediapipe numpy pygame`
2: Run
3: Show your hands towards the camera ( I do recommand you to watch the sample video inside this repository first) . Use your left fingers to make gestures responding to different sounds, while slowly wave your right hand to play. Raise your left thumb to pause the sound instantly. 


Here is the music score of the exact same piece I performed in the sample video. It's a small fragment of minecraft's famouse background music "mice on venus" . 
1111 (fa)
1110 (mi)
1000 (do)
1000 (do)
1000 (do)
1110 (mi)
1000 (do)
1110 (mi)
0111 (so)
1001 (re_high)
1101 (mi_high)
1001 (re_high)
0101 (so_high)
0001 (do_high)
0011 (la)
1101 (mi_high)
1001 (re_high)
0001 (do_high)
0001 (do_high)
0001 (do_high)
0011 (la)
0111 (so)
1111 (fa)
0111 (so)
1110 (mi)


   
