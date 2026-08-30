# AI Task 3 — Music Generation with AI (LSTM)

An LSTM-based deep learning model that learns musical patterns from a MIDI
dataset (classical, jazz, etc.) and generates new original music sequences,
which are then converted back into a playable MIDI file.

## Pipeline

1. **Data collection** — MIDI files (classical/jazz) collected as training data
2. **Preprocessing** — notes and chords extracted from MIDI files using `music21`
3. **Sequence preparation** — notes converted into fixed-length input/output sequences
4. **Model** — a 3-layer LSTM network (deep learning) built with TensorFlow/Keras
5. **Training** — model trained to predict the next note given a sequence of previous notes
6. **Generation** — trained model used to generate a brand-new sequence of notes
7. **Export** — generated sequence converted back into a `.mid` file (and optionally `.wav` audio)

## How to Run (Google Colab)

1. Open a new Colab notebook and enable GPU: `Runtime > Change runtime type > GPU`
2. In the first cell:
   ```
   !pip install music21 midi2audio
   !apt-get install -y fluidsynth
   ```
3. Upload `music_generation.py`.
4. Create a `midi_songs/` folder and upload your MIDI (`.mid`) files into it
   (pick one genre/style, at least 30-50 files for decent results).
5. Train the model:
   ```
   !python music_generation.py --mode train --epochs 100
   ```
6. Generate new music (after training):
   ```
   !python music_generation.py --mode generate
   ```
   This produces `generated_output.mid`. To also get a `.wav` audio file, run
   the two lines printed at the end of the script's output.

## Notes

- Training on CPU is very slow — a GPU (free tier on Colab) is strongly recommended.
- The larger and more stylistically consistent the dataset, the better the output.
- `SEQUENCE_LENGTH`, `epochs`, and the number of LSTM units (inside the script)
  can be tuned for better results.

## Tech Stack

- Python
- [music21](https://web.mit.edu/music21/) — MIDI parsing and music theory objects
- TensorFlow / Keras — LSTM deep learning model
- FluidSynth / midi2audio — MIDI-to-audio conversion (optional)
