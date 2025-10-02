"""
inference.py - Music Transcription and Instrument Identification Inference Script

This script performs inference on audio files using a pre-trained multi-task Keras model
that performs both music transcription (piano roll generation) and instrument identification.
"""

import os
import numpy as np
import tensorflow as tf
import librosa
import pretty_midi
from train import build_model,INSTRUMENT_MAP

# Hard-coded parameters matching the training script
SAMPLE_RATE = 16000
DURATION = 10.0
HOP_LENGTH = 256
N_PITCHES = 88
MIDI_PITCH_OFFSET = 21

def load_and_preprocess_audio(audio_path):
    """
    Load and preprocess audio file for model inference.
    
    Args:
        audio_path (str): Path to the audio file
        
    Returns:
        np.ndarray: Preprocessed audio array with shape (1, SAMPLE_RATE * DURATION)
    """
    # Load audio as mono and resample to target sample rate
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    
    # Fix length to exactly DURATION * SAMPLE_RATE samples
    target_length = int(DURATION * SAMPLE_RATE)
    audio = librosa.util.fix_length(audio, size=target_length)
    
    # Add batch dimension
    audio = np.expand_dims(audio, axis=0)
    
    return audio


def piano_roll_to_midi(piano_roll, output_path='transcribed_output.mid'):
    """
    Convert a binary piano roll to a MIDI file.
    
    Args:
        piano_roll (np.ndarray): Binary piano roll of shape (n_frames, n_pitches)
        output_path (str): Path to save the MIDI file
        
    Returns:
        pretty_midi.PrettyMIDI: The created MIDI object
    """
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    
    # Create an instrument instance (using piano by default)
    instrument = pretty_midi.Instrument(program=0)
    
    # Calculate frame duration in seconds
    frame_duration = HOP_LENGTH / SAMPLE_RATE
    
    # Iterate through each pitch
    for pitch_idx in range(N_PITCHES):
        # Get the activation for this pitch across all frames
        pitch_activations = piano_roll[:, pitch_idx]
        
        # Find note on/off events
        note_on = False
        start_frame = 0
        
        for frame_idx in range(len(pitch_activations)):
            if pitch_activations[frame_idx] and not note_on:
                # Note onset detected
                note_on = True
                start_frame = frame_idx
            elif not pitch_activations[frame_idx] and note_on:
                # Note offset detected
                note_on = False
                
                # Convert frame indices to time
                start_time = start_frame * frame_duration
                end_time = frame_idx * frame_duration
                
                # Calculate MIDI pitch number
                midi_pitch = pitch_idx + MIDI_PITCH_OFFSET
                
                # Create note object
                note = pretty_midi.Note(
                    velocity=80,
                    pitch=midi_pitch,
                    start=start_time,
                    end=end_time
                )
                
                # Add note to instrument
                instrument.notes.append(note)
        
        # Handle case where note extends to the end
        if note_on:
            start_time = start_frame * frame_duration
            end_time = len(pitch_activations) * frame_duration
            midi_pitch = pitch_idx + MIDI_PITCH_OFFSET
            
            note = pretty_midi.Note(
                velocity=80,
                pitch=midi_pitch,
                start=start_time,
                end=end_time
            )
            instrument.notes.append(note)
    
    # Add instrument to MIDI object
    midi.instruments.append(instrument)
    
    # Save MIDI file
    midi.write(output_path)
    
    return midi


def perform_inference(audio_path, model_weights_path):
    """
    Main inference function that loads the model, processes audio, and generates outputs.
    
    Args:
        audio_path (str): Path to the input audio file
        model_weights_path (str): Path to the saved model weights (.h5 file)
    """
    print(f"Loading audio from: {audio_path}")
    
    # Load and preprocess audio
    audio = load_and_preprocess_audio(audio_path)
    print(f"Audio preprocessed: shape {audio.shape}")
    
    # Build model architecture
    print("Building model architecture...")
    # Build model architecture
    print("Building model architecture...")
    # Calculate the shape parameters the model needs
    input_samples = int(DURATION * SAMPLE_RATE)
    n_frames = input_samples // HOP_LENGTH
    n_pitches = N_PITCHES
    n_instruments = len(INSTRUMENT_MAP)

    # Pass the parameters to the model builder
    model = build_model(
        input_shape=(input_samples,),
        n_frames=n_frames,
        n_pitches=n_pitches,
        n_instruments=n_instruments
    )
    print("Model architecture built.")
    
    # Load pre-trained weights
    print(f"Loading model weights from: {model_weights_path}")
    model.load_weights(model_weights_path)
    
    # Perform inference
    print("Running inference...")
    predictions = model.predict(audio)
    
    # Process instrument prediction
    instrument_probs = predictions['instruments'][0]  # Remove batch dimension
    predicted_instrument_idx = np.argmax(instrument_probs)
    predicted_instrument = [key for key, value in INSTRUMENT_MAP.items() if value == predicted_instrument_idx][0]
    confidence = instrument_probs[predicted_instrument_idx]
    
    predicted_instrument = list(INSTRUMENT_MAP.keys())[predicted_instrument_idx]
    
    print(f"\n=== Instrument Identification ===")
    print(f"Predicted Instrument: {predicted_instrument}")
    print(f"Confidence: {confidence:.2%}")
    
    # Process note predictions (piano roll)
    note_probs = predictions['notes'][0]  # Remove batch dimension
    note_probs = np.squeeze(note_probs, axis=-1) 
    binary_piano_roll = (note_probs > 0.5).astype(np.float32)
    
    # Count active notes
    n_active_notes = np.sum(binary_piano_roll)
    n_frames, n_pitches = binary_piano_roll.shape
    
    print(f"\n=== Music Transcription ===")
    print(f"Piano roll shape: {binary_piano_roll.shape}")
    print(f"Total active notes: {int(n_active_notes)}")
    print(f"Frames: {n_frames}, Pitches: {n_pitches}")
    
    # Convert to MIDI and save
    output_midi_path = 'transcribed_output.mid'
    print(f"\nConverting to MIDI and saving to: {output_midi_path}")
    midi_data = piano_roll_to_midi(binary_piano_roll, output_midi_path)
    
    # Print MIDI statistics
    total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
    print(f"MIDI file created with {total_notes} notes")
    print(f"Duration: {midi_data.get_end_time():.2f} seconds")
    
    print("\nInference completed successfully!")


if __name__ == '__main__':
    # User-configurable paths
    TEST_AUDIO_PATH = "C:/Users/Avi Pandey/Downloads/Music/Easy Piano Tutorial- Twinkle Twinkle Little Star.mp3" # Path to your test audio file
    MODEL_WEIGHTS_PATH = "pit_net_model_final.weights.h5"  # Path to your saved model weights
    
    # Check if files exist
    if not os.path.exists(TEST_AUDIO_PATH):
        print(f"Error: Audio file not found at {TEST_AUDIO_PATH}")
        print("Please update TEST_AUDIO_PATH with the correct path to your audio file.")
        exit(1)
    
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: Model weights file not found at {MODEL_WEIGHTS_PATH}")
        print("Please update MODEL_WEIGHTS_PATH with the correct path to your model weights.")
        exit(1)
    
    # Check if train.py exists
    if not os.path.exists("train.py"):
        print("Error: train.py not found in the current directory.")
        print("Please ensure train.py is in the same directory as this script.")
        exit(1)
    
    # Run inference
    try:
        perform_inference(TEST_AUDIO_PATH, MODEL_WEIGHTS_PATH)
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)