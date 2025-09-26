# slakh_dataset.py (Final Version)
import os
import yaml
import numpy as np
import librosa
import tensorflow as tf
from typing import Dict, Tuple

INSTRUMENT_MAP = {
    'Piano': 0, 'Chromatic Percussion': 1, 'Organ': 2, 'Guitar': 3, 'Bass': 4,
    'Strings': 5, 'Ensemble': 6, 'Brass': 7, 'Reed': 8, 'Pipe': 9, 'Synth Lead': 10,
    'Synth Pad': 11, 'Synth Effects': 12, 'Ethnic': 13, 'Percussive': 14, 'Sound Effects': 15, 'Drums': 16,
}

def create_piano_roll(notes: list, n_frames: int, sr: int, hop_length: int, n_pitches: int = 88, midi_offset: int = 21):
    piano_roll = np.zeros((n_frames, n_pitches), dtype=np.float32)
    for note in notes:
        start_time = note['onset_sec']
        end_time = note['offset_sec']
        pitch = note['pitch_midi']
        start_frame = int(np.floor(start_time * sr / hop_length))
        end_frame = int(np.ceil(end_time * sr / hop_length))
        note_idx = pitch - midi_offset
        if 0 <= note_idx < n_pitches:
            piano_roll[start_frame:end_frame, note_idx] = 1.0
    return piano_roll

def load_slakh_track_data(track_path: str, sample_rate: int, duration: float, hop_length: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    mix_path = os.path.join(track_path, 'mix.flac')
    audio, sr = librosa.load(mix_path, sr=sample_rate, mono=True, duration=duration)
    
    target_samples = int(duration * sample_rate)
    audio = librosa.util.fix_length(audio, size=target_samples)
    
    metadata_path = os.path.join(track_path, 'metadata.yaml')
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
        
    instrument_label = np.zeros(len(INSTRUMENT_MAP), dtype=np.float32)
    for stem_info in metadata.get('stems', {}).values():
        inst_class = stem_info.get('inst_class')
        if inst_class in INSTRUMENT_MAP:
            instrument_label[INSTRUMENT_MAP[inst_class]] = 1.0
            
    n_frames = len(audio) // hop_length
    all_notes = [note for stem in metadata.get('stems', {}).values() for note in stem.get('notes', [])]
    
    note_matrix = create_piano_roll(all_notes, n_frames, sample_rate, hop_length)
    onset_matrix = np.zeros_like(note_matrix) # Placeholder
    contour_matrix = np.zeros_like(note_matrix) # Placeholder simplified to 88 bins

    output_labels = {
        "notes": note_matrix, "onsets": onset_matrix,
        "contours": contour_matrix, "instruments": instrument_label,
    }
    return audio, output_labels

class SlakhDataset:
    def __init__(self, data_dir: str, split: str, sample_rate: int, duration: float, hop_length: int):
        self.split_dir = os.path.join(data_dir, split)
        self.track_dirs = sorted([os.path.join(self.split_dir, d) for d in os.listdir(self.split_dir) if d.startswith('Track')])
        self.sample_rate = sample_rate
        self.duration = duration
        self.hop_length = hop_length
        
    def __len__(self):
        return len(self.track_dirs)
    
    def __getitem__(self, idx):
        return load_slakh_track_data(self.track_dirs[idx], self.sample_rate, self.duration, self.hop_length)
        
    def create_tf_dataset(self, batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
        def generator():
            indices = list(range(len(self)))
            if shuffle: np.random.shuffle(indices)
            for idx in indices:
                yield self[idx]
                
        n_frames = int(self.duration * self.sample_rate) // self.hop_length
        output_signature = (
            tf.TensorSpec(shape=(int(self.duration * self.sample_rate),), dtype=tf.float32),
            {
                "notes": tf.TensorSpec(shape=(n_frames, 88), dtype=tf.float32),
                "onsets": tf.TensorSpec(shape=(n_frames, 88), dtype=tf.float32),
                "contours": tf.TensorSpec(shape=(n_frames, 88), dtype=tf.float32),
                "instruments": tf.TensorSpec(shape=(len(INSTRUMENT_MAP),), dtype=tf.float32),
            }
        )
        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset