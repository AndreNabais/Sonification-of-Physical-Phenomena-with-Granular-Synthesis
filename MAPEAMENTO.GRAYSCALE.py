from PIL import Image
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
from scipy.ndimage import gaussian_filter1d
from pythonosc.udp_client import SimpleUDPClient
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import cm
from scipy.spatial import KDTree

# ===================== SETUP =====================
image_path = r"c:\Users\andre\Midi_PIC\imagens\image4_grayscale.png"
midi_output_path = r"c:\Users\andre\Midi_PIC\Código_Teste\image4_grayscale_3freq.mid"

client = SimpleUDPClient("127.0.0.1", 57120)

note_range = (48, 72)
top_n = 3
threshold = 0.2
time_per_line = 0.02
sigma = 2
# =================================================

# ===== LOAD IMAGE + ORIGINAL KDTree COLOR MAPPING =====
img = Image.open(image_path).convert('RGB')
img_array = np.array(img)
img_array = np.flipud(img_array)

colormap = cm.get_cmap('jet', 256)
cmap_colors = (colormap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
tree = KDTree(cmap_colors)

pixels = img_array.reshape(-1, 3)
_, indices = tree.query(pixels)
intensity = indices / (len(cmap_colors) - 1)
intensity_image = intensity.reshape(img_array.shape[0], img_array.shape[1])
# =================================================

# ===== MIDI Note Mapping Across Image Width =====
midi_notes = np.linspace(note_range[0], note_range[1], img_array.shape[1]).astype(int)

mid = mido.MidiFile()
track = MidiTrack()
mid.tracks.append(track)

tempo = mido.bpm2tempo(120)
ticks_per_beat = 480
track.append(MetaMessage('set_tempo', tempo=tempo))

def seconds_to_ticks(seconds):
    return int(mido.second2tick(seconds, ticks_per_beat, tempo))

# ========= ANALYSIS AND MIDI GENERATION ==========
times, notes = [], []

for row_idx, row in enumerate(intensity_image):
    row = gaussian_filter1d(row, sigma=sigma)
    current_time = row_idx * time_per_line
    notes_in_line = []

    top_indices = np.argsort(row)[-top_n:]
    max_col = max(top_indices)

    for col_idx in top_indices:
        intensity = row[col_idx]
        if intensity > threshold:
            note = midi_notes[col_idx]
            velocity = int(intensity * 127)
            notes_in_line.append((note, velocity))
            times.append(current_time)
            notes.append(note)

    for note, velocity in notes_in_line:
        track.append(Message('note_on', note=note, velocity=velocity, time=0))

    for i, (note, velocity) in enumerate(notes_in_line):
        t = seconds_to_ticks(time_per_line) if i == 0 else 0
        track.append(Message('note_off', note=note, velocity=velocity, time=t))
# ===================================================

# ============== VISUALIZATION ======================
plt.figure(figsize=(12, 6))
plt.scatter(times, notes, s=5, c='blue')
plt.xlabel("Tempo (s)")
plt.ylabel("Nota MIDI")
plt.title("Gráfico Tempo vs Nota (mapeamento do espetrograma)")
plt.grid(True)
plt.show()
# ===================================================

# ================ SAVE MIDI FILE ===================
mid.save(midi_output_path)
print(f"MIDI file saved: {midi_output_path}")
print("Image shape:", intensity_image.shape)
# ===================================================

# ========== OSC PARAMETER STREAM TO SC ============
for row in intensity_image:
    row = gaussian_filter1d(row, sigma=sigma)

    derivative = np.abs(np.diff(row))
    avg_derivative = np.mean(derivative)
    randomness = min(avg_derivative * 2.0, 0.02)

    max_intensity = np.max(row)
    amp = max_intensity
    max_col = np.argmax(row)
    pos = max_col / len(row)
    spread = np.std(row)

    center_col = img_array.shape[1] // 2
    distance_from_center = max_col - center_col
    pan = np.clip(distance_from_center / (img_array.shape[1] / 2), -1.0, 1.0)

    top_indices = np.argsort(row)[-top_n:]
    rate = np.mean(top_indices)

        # Map dominant note to pitch parameter (normalized from MIDI note)
    dominant_note = midi_notes[max_col]
    pitch = (dominant_note - note_range[0]) / (note_range[1] - note_range[0])  # normalize to 0–1

    client.send_message("/grain/pos", pos)
    client.send_message("/grain/amp", amp)
    client.send_message("/grain/jitter", randomness)
    client.send_message("/grain/spread", spread)
    client.send_message("/grain/pan", pan)
    client.send_message("/grain/rate", rate)
    client.send_message("/grain/pitch", pitch)

    time.sleep(time_per_line)
# ===================================================
# NOW stop the synth AFTER the loop ends
client.send_message("/grain/stop", 1)