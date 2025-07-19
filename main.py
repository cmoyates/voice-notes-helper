from openai import OpenAI
from dotenv import load_dotenv
import pyaudio
import wave
import threading
from faster_whisper import WhisperModel
import time

# Load environment variables from .env file
load_dotenv()


model_size = "small"
OUTPUT_FILENAME = "output.wav"


def record_until_keypress(
    output_filename=OUTPUT_FILENAME,
    chunk=1024,
    format=pyaudio.paInt16,
    channels=2,
    rate=44100,
):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk
    )
    frames = []
    stop_flag = threading.Event()

    def _record():
        print("Recording... Press Enter to stop")
        while not stop_flag.is_set():
            data = stream.read(chunk)
            frames.append(data)
        print("Stopped")

    threading.Thread(target=_record, daemon=True).start()
    input()  # Wait for Enter keypress
    stop_flag.set()
    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(output_filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))

    print(f"Saved to {output_filename}")


def transcribe_audio(file_path):
    # Initialize model inside function to avoid segmentation faults
    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
    )

    segments, info = model.transcribe(
        file_path,
        beam_size=5,
        language="en",
    )
    transcript = ""
    for segment in segments:
        transcript += f"{segment.text} "
    return transcript.strip()


def main():
    try:
        # record_until_keypress()
        time.sleep(2)

        transcript = transcribe_audio(OUTPUT_FILENAME)

        print("Transcription:")
        print(transcript)

        client = OpenAI()

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that cleans up transcriptions. Please remove any unnecessary filler words, pauses, or repetitions from the transcription.",
                },
                {
                    "role": "user",
                    "content": f"Please clean up the following transcription:\n\n{transcript}",
                },
            ],
        )

        fixed_transcript = (
            completion.choices[0].message.content.strip()
            if completion.choices[0].message.content
            else "No response from model"
        )

        print()

        print("Fixed Transcript:")
        print(fixed_transcript)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
