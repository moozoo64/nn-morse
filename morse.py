#!/usr/bin/env python3

from scipy import signal
import numpy as np
import random

SAMPLE_FREQ = 4000  # 2 Khz

MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...',
    'C': '-.-.', 'D': '-..', 'E': '.',
    'F': '..-.', 'G': '--.', 'H': '....',
    'I': '..', 'J': '.---', 'K': '-.-',
    'L': '.-..', 'M': '--', 'N': '-.',
    'O': '---', 'P': '.--.', 'Q': '--.-',
    'R': '.-.', 'S': '...', 'T': '-',
    'U': '..-', 'V': '...-', 'W': '.--',
    'X': '-..-', 'Y': '-.--', 'Z': '--..',
    '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....',
    '7': '--...', '8': '---..', '9': '----.',
    '0': '-----',
    '.': '.-.-.-', ',': '--..--', '?': '..--..',
    '=': '-...-', '+': '.-.-.',
}
ALPHABET = " " + "".join(MORSE_CODE_DICT.keys())


def get_spectrogram(samples):
    window_length = int(0.02 * SAMPLE_FREQ)  # 20 ms windows
    _, _, s = signal.spectrogram(samples, nperseg=window_length, noverlap=0)
    return s

def loadQRMS():
    import os
    from scipy.io.wavfile import read
    qrms = []
    filenames = os.listdir("qrm")
    filenames.sort()
    for filename in filenames:
            filename="qrm/"+filename
            qrm=read(filename)[1]
            qrms.append(qrm)
    return qrms
QRMS = loadQRMS()

def generate_sample(text_len=10, pitch=500, wpm=20, noise_power=1, qrmIndex=0, fadePeriod=0,fadePhase=0, amplitude=100, s=None):
    assert pitch < SAMPLE_FREQ / 2  # Nyquist

    # Reference word is PARIS, 50 dots long
    dot = (60 / wpm) / 50 * SAMPLE_FREQ

    # Add some noise on the length of dash and dot
    def get_dot():
        scale = np.clip(np.random.normal(1, 0.2), 0.5, 2.0)
        return int(dot * scale)

    # The length of a dash is three times the length of a dot.
    def get_dash():
        scale = np.clip(np.random.normal(1, 0.2), 0.5, 2.0)
        return int(3 * dot * scale)

    # Create random string that doesn't start or end with a space
    #if s is None:
    #    s1 = ''.join(random.choices(ALPHABET, k=text_len - 2))
    #    s2 = ''.join(random.choices(ALPHABET[1:], k=2))
    #   s = s2[0] + s1 + s2[1]

    # Create random string of words between 1 and 6 characters long that doesn't start or end with a space
    s=""
    while len(s)<text_len:
        word_lenght = text_len+10
        while word_lenght+len(s) > text_len:
            word_lenght = random.randrange(1, 6)
        word =  ''.join(random.choices(ALPHABET[1:], k=word_lenght))
        if(len(s)+word_lenght+1<text_len):
            s=s+word+" "
        else:
            s=s+word

    out = []
    out.append(np.zeros(5 * get_dot()))

    # The space between two signs of the same character is equal to the length of one dot.
    # The space between two characters of the same word is three times the length of a dot.
    # The space between two words is seven times the length of a dot (or more).
    for c in s:
        if c == ' ':
            out.append(np.zeros(7 * get_dot()))
        else:
            for m in MORSE_CODE_DICT[c]:
                if m == '.':
                    out.append(np.ones(get_dot()))
                    out.append(np.zeros(get_dot()))
                elif m == '-':
                    out.append(np.ones(get_dash()))
                    out.append(np.zeros(get_dot()))

            out.append(np.zeros(2 * get_dot()))

    out.append(np.zeros(5 * get_dot()))
    out = np.hstack(out)

    # Modulatation
    t = np.arange(len(out)) / SAMPLE_FREQ
    sine = np.sin(2 * np.pi * t * pitch)
    if(fadePeriod!=0):
        sine = sine * (0.7+0.3*np.sin(np.pi * t / fadePeriod + fadePhase))
    out = sine * out

    # Add noise
    if noise_power != 0:
        noise_power = 1e-6 * noise_power * SAMPLE_FREQ / 2
        noise = np.random.normal(scale=np.sqrt(noise_power), size=len(out))
    
    if qrmIndex !=0 :
        qrmFile=QRMS[qrmIndex]
        qrm=qrmFile
        while len(qrm)<len(out):
            qrm=np.concatenate((qrm,qrmFile))
        qrm=qrm[:len(out)]
        qrm = qrm/np.sqrt(np.mean(qrm**2)) * 0.5
        if noise_power != 0:
            out = 0.5 * out + 0.45*noise + 0.05*qrm
        else:
            out = 0.5 * out + 0.05*qrm
    else:
        if noise_power !=0:
            out = 0.5 * out + noise
        else:
            out = 0.5 * out

    out *= amplitude / 100
    out = np.clip(out, -1, 1)

    out = out.astype(np.float32)

    spec = get_spectrogram(out)

    return out, spec, s


if __name__ == "__main__":
    from scipy.io.wavfile import write
    import matplotlib.pyplot as plt
    import os

    length = random.randrange(10, 20)
    pitch = random.randrange(100, 950)
    wpm = random.randrange(10, 40)
    noise_power = random.randrange(0, 200)
    amplitude = random.randrange(10, 150)
    qrmIndex = random.randrange(0, len(os.listdir("QRM")))
    fadePeriod = random.randrange(0, 10)
    fadePhase = 2*np.pi*random.randrange(0, 100)/100.0

    s = "HELLO, WORLD"
    samples, spec, y = generate_sample(length, pitch, wpm, noise_power,qrmIndex, fadePeriod,fadePhase, amplitude, s)
    samples = samples.astype(np.float32)
    write("morse.wav", SAMPLE_FREQ, samples)

    dotSamples = int((60 / wpm) / 50 * SAMPLE_FREQ)
    print(f"pitch: {pitch} wpm: {wpm} dotTime: {dotSamples} noise: {noise_power} qrmIndex: {qrmIndex} fadePeriod: {fadePeriod} fadePhase: {fadePhase} amplitude: {amplitude} {y}")

    plt.figure()
    plt.pcolormesh(spec)
    plt.show()
