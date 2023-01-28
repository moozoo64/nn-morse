import numpy as np
import ffmpeg
import os
from scipy.io.wavfile import write
SAMPLE_FREQ = 2000  # 2 Khz

def load_audio(file: str, sample_rate: int = SAMPLE_FREQ):
    try:
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sample_rate)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

if __name__ == "__main__":    
    filenames = os.listdir("QRMin")
    filenames.sort()
    for filename in filenames:
        qrm=load_audio("QRMin/"+filename)
        filenameOut=filename.split(".")[0]+".wav"
        print(filenameOut)
        write("qrm/"+filenameOut, SAMPLE_FREQ, qrm)