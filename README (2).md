# AI Task 3 — Music Generation with AI (LSTM)

## Colab par chalane ka tareeqa

1. Naya Colab notebook banayein aur GPU on karein: `Runtime > Change runtime type > GPU`
2. Pehle cell mein:
   ```
   !pip install music21 midi2audio
   !apt-get install -y fluidsynth
   ```
3. `music_generation.py` upload karein (ya cell mein paste kar dein).
4. `midi_songs/` folder banayein aur usmein apni MIDI (`.mid`) files upload karein
   (classical/jazz — jo bhi dataset chunein, kam se kam 30-50 files).
5. Training:
   ```
   !python music_generation.py --mode train --epochs 100
   ```
6. Generation (training ke baad):
   ```
   !python music_generation.py --mode generate
   ```
   Ye `generated_output.mid` bana dega. Audio (.wav) ke liye script ke end mein
   diye gaye 2 lines run kar lein.

## Notes
- CPU par training bohat slow hogi — GPU (Colab free tier) strongly recommended.
- Dataset jitni bari aur consistent (ek hi genre/style) hogi, output utna behtar hoga.
- `SEQUENCE_LENGTH`, `epochs`, aur LSTM units (script ke andar) experiment kar ke tune kar sakte hain.
