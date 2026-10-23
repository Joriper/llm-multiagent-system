
import asyncio
import websockets
import speech_recognition as sr

r = sr.Recognizer()
mic = sr.Microphone()

def listen_mic():

    source = mic.__enter__()
    r.adjust_for_ambient_noise(source, duration=1)
    print("Speak something...")
    audio = r.listen(source,timeout=10)
    mic.__exit__(None, None, None)

    try:
        text = r.recognize_google(audio, language="en-IN")
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Voice not understood")
        return None


async def main():
    ws = await websockets.connect("ws://127.0.0.1:8000/ws/ai",    ping_interval=None,
    open_timeout=30)
    while True:
        input("Press Enter to speak...")

        text = listen_mic()
        if text:
            await ws.send(text)
            response = await ws.recv()
            print("AI Response:", response)


asyncio.run(main())
