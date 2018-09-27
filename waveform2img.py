import argparse
import glob
import os
import re
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import butter, lfilter

def get_emotion_label(filename):
    """
    Each of the 7356 RAVDESS files has a unique filename. The filename consists of a 7-part numerical identifier
    (e.g., 02-01-06-01-02-01-12.mp4). These identifiers define the stimulus characteristics:

    Filename identifiers

    Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    Vocal channel (01 = speech, 02 = song).
    Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
    Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
    Repetition (01 = 1st repetition, 02 = 2nd repetition).
    Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

    Here we will only use 'Emotion' as the label for our training

    INPUT
        filename

    OUTPUT
        emotion label, a string
    """
    labels_dict = dict(zip(range(8),
                           ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']))

    emotion_id_pos = 2
    label_ind = int(re.findall(r"\d+", os.path.basename(filename))[emotion_id_pos]) - 1
    return labels_dict[label_ind]


# Define a function which wil apply a butterworth bandpass filter


def butter_bandpass_filter(samples, sample_rate, lowcut=30, highcut=3000, order=5):
    """
    Butterworth's filter
    """

    def butter_bandpass(lowcut, highcut, sample_rate, order=5):
        nyq = 0.5 * sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    b, a = butter_bandpass(lowcut, highcut, sample_rate, order=order)
    y = lfilter(b, a, samples)
    return y


def get_melspectrogram(samples, sample_rate):
    """
    return a normalized spectrogram of type uint8

    INPUT
        samples
        sample_rate
    OUTPUT
        spectrogram   2D array, where axis 0 is time and axis 1 is fourier decomposition
                      of waveform at different times
    """
    melspectrogram = librosa.feature.melspectrogram(samples, sample_rate)

    # max L-infinity normalized the energy
    normalized = librosa.util.normalize(melspectrogram)

    # scale to 8-bit representation
    scaled = 255 * normalized  # Now scale by 255
    return scaled.astype(np.uint8)


def align(spectrogram, target_frames=128):
    """
    Align all input to the same length (default is 128 frames).
    """
    _, n_frames = spectrogram.shape

    if n_frames < target_frames:
        npad = ((0, 0), (0, target_frames - n_frames), (0, 0))
        return np.pad(spectrogram, pad_width=npad, mode='wrap')

    return spectrogram[:, :target_frames]


def duplicate_and_stack(layer, dups=3):
    """
    Images used for training should contain 3 channels. This function
    duplicates the 2D array 3 times to conform to this requirement.
    :param layer:
    :param dups:
    :return:
    """
    return np.stack((layer for _ in range(dups)), axis=2)


N_MINIMUM_SAMPLES = 30000  # samples with lower than 30000 frames usually do not contain utterances so we drop them


def get_save_path(output_root, audio_file):
    """
    Compute save path for a given audio file based on its emotion label
    Image representation should be saved in a subfolder indexed by its label
    :param output_root:
    :param audio_file: name of audio file, contains the emotion identifier
    :return:
    """
    def get_or_create_subfolder():
        out_dir = Path(output_root) / Path(get_emotion_label(audio_file))
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    output_dir = get_or_create_subfolder()
    output_file_name = os.path.splitext(os.path.basename(audio_file))[0] + '.jpg'

    return os.path.join(output_dir, output_file_name)


def convert(input_name_pattern, output_root):
    # takes about 6-8 min on my machine
    counter = 0
    print(input_name_pattern)
    for audio_file in glob.iglob(input_name_pattern, recursive=True):
        samples, sample_rate = librosa.load(audio_file)

        if len(samples) > N_MINIMUM_SAMPLES:
            samples = butter_bandpass_filter(samples, sample_rate)
            spectrogram = get_melspectrogram(samples, sample_rate)

            # obtain labels, get or create sub-folder
            wav_img_save_path = get_save_path(output_root, audio_file);

            # save spectrogram to sub-folder
            img = Image.fromarray(duplicate_and_stack(spectrogram))
            img.save(wav_img_save_path)

            if counter % 100 == 0:
                print('Processing the {}th file: {}'.format(counter, audio_file))
            counter += 1


def main():
    parser = argparse.ArgumentParser(description='Convert input waveforms into image '
                                                 'representations and save them under folders '
                                                 'named after their labels')
    parser.add_argument('input_dir',
                        help='Directory that contains all the input files. Can contain nested folders.')
    parser.add_argument('-p',
                        '--pattern',
                        default='**/*seg0.wav',
                        help='Pattern used in searching sound files')
    parser.add_argument('output_dir',
                        default='wav2img/',
                        help='Directory to save output imgs to. Default to wav2img/')

    args = parser.parse_args()
    input_name_pattern = str(Path(args.input_dir) / Path(args.pattern))
    convert(input_name_pattern, args.output_dir)


if __name__ == '__main__':
    main()
