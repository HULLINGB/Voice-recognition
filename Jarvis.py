import pyttsx3
import speech_recognition as sr
#for windows computers use pyaudio for speech-to-text
import pyaudio as pya

r = sr.Recognizer()
#r2 = pya.PyAudio()
with sr.Microphone() as source:
    print("Speak...")
    audio = r.listen(source)
print(r.recognize_google(audio))
if(r.recognize_google(audio)) == "Jarvis":
    engine = pyttsx3.init()
    engine.say("Hello Mr Stark")
    engine.runAndWait()