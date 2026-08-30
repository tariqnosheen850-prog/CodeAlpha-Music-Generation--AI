"""
AI Task 3: Music Generation with AI (LSTM + music21)
======================================================
Ye script Google Colab ya local machine par chal sakti hai.

SETUP (Colab mein pehle cell mein run karein):
    !pip install music21 midi2audio
    !apt-get install -y fluidsynth   # sirf audio (.wav) export ke liye chahiye

DATASET:
    Ek folder "midi_songs/" banayein aur usmein apni MIDI (.mid) files daal dein
    (classical/jazz — jo bhi genre chahein). Colab mein left sidebar se upload
    kar sakte hain, ya Google Drive mount kar ke wahan se path dein.

USAGE:
    python music_generation.py --mode train      # dataset se model train karega
    python music_generation.py --mode generate   # trained weights se naya music banayega
"""

import argparse
import glob
import pickle
import os
import numpy as np

from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

MIDI_FOLDER = "midi_songs"
NOTES_FILE = "data/notes"
SEQUENCE_LENGTH = 100
WEIGHTS_FILE = "best_weights.hdf5"
OUTPUT_MIDI = "generated_output.mid"


# ---------------------------------------------------------------------
# STEP 1 + 2: MIDI files parse karna aur notes/chords extract karna
# ---------------------------------------------------------------------
def get_notes():
    os.makedirs("data", exist_ok=True)
    notes = []

    midi_files = glob.glob(os.path.join(MIDI_FOLDER, "*.mid"))
    if not midi_files:
        raise FileNotFoundError(
            f"'{MIDI_FOLDER}/' mein koi .mid file nahi mili. Pehle dataset daalein."
        )

    for file in midi_files:
        print(f"Parsing: {file}")
        try:
            midi = converter.parse(file)
        except Exception as e:
            print(f"  Skip kar rahe hain (corrupt/unsupported file): {e}")
            continue

        parts = instrument.partitionByInstrument(midi)
        notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(n) for n in element.normalOrder))

    with open(NOTES_FILE, "wb") as f:
        pickle.dump(notes, f)

    print(f"Total notes/chords extracted: {len(notes)}")
    return notes


# ---------------------------------------------------------------------
# STEP 3: Sequences banana (training format)
# ---------------------------------------------------------------------
def prepare_sequences(notes, n_vocab):
    pitchnames = sorted(set(notes))
    note_to_int = {n: i for i, n in enumerate(pitchnames)}

    network_input = []
    network_output = []

    for i in range(0, len(notes) - SEQUENCE_LENGTH):
        seq_in = notes[i:i + SEQUENCE_LENGTH]
        seq_out = notes[i + SEQUENCE_LENGTH]
        network_input.append([note_to_int[c] for c in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    network_input_raw = network_input  # generation ke liye raw (un-normalized) copy

    X = np.reshape(network_input, (n_patterns, SEQUENCE_LENGTH, 1))
    X = X / float(n_vocab)
    y = to_categorical(network_output)

    return X, y, pitchnames, note_to_int, network_input_raw


# ---------------------------------------------------------------------
# STEP 4: LSTM Model
# ---------------------------------------------------------------------
def create_model(input_shape, n_vocab):
    model = Sequential()
    model.add(LSTM(512, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


# ---------------------------------------------------------------------
# STEP 5: Training
# ---------------------------------------------------------------------
def train(epochs=100, batch_size=64):
    notes = get_notes()
    n_vocab = len(set(notes))
    X, y, pitchnames, note_to_int, _ = prepare_sequences(notes, n_vocab)

    model = create_model((X.shape[1], X.shape[2]), n_vocab)
    checkpoint = ModelCheckpoint(WEIGHTS_FILE, monitor="loss", save_best_only=True, mode="min")

    print(f"Training shuru — {n_vocab} unique notes/chords, {len(X)} sequences.")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])
    print(f"Training mukammal. Best weights '{WEIGHTS_FILE}' mein save hain.")


# ---------------------------------------------------------------------
# STEP 6 + 7: Generation aur MIDI mein convert karna
# ---------------------------------------------------------------------
def generate(num_notes=200):
    with open(NOTES_FILE, "rb") as f:
        notes = pickle.load(f)

    n_vocab = len(set(notes))
    X, y, pitchnames, note_to_int, network_input_raw = prepare_sequences(notes, n_vocab)
    int_to_note = {i: n for n, i in note_to_int.items()}

    model = create_model((X.shape[1], X.shape[2]), n_vocab)
    model.load_weights(WEIGHTS_FILE)

    start = np.random.randint(0, len(network_input_raw) - 1)
    pattern = list(network_input_raw[start])
    prediction_output = []

    for _ in range(num_notes):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        prediction_output.append(int_to_note[index])
        pattern.append(index)
        pattern = pattern[1:]

    create_midi(prediction_output)


def create_midi(prediction_output):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ("." in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split(".")
            chord_notes = []
            for current_note in notes_in_chord:
                n = note.Note(int(current_note))
                n.storedInstrument = instrument.Piano()
                chord_notes.append(n)
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            n = note.Note(pattern)
            n.offset = offset
            n.storedInstrument = instrument.Piano()
            output_notes.append(n)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp=OUTPUT_MIDI)
    print(f"Generated MIDI saved: {OUTPUT_MIDI}")
    print("Audio (.wav) chahiye to ye run karein:")
    print("  from midi2audio import FluidSynth")
    print(f"  FluidSynth().midi_to_audio('{OUTPUT_MIDI}', 'generated_output.wav')")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "generate"], required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "train":
        train(epochs=args.epochs)
    else:
        generate()
