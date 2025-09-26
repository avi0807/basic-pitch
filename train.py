# train.py (Final Version)
import os
import tensorflow as tf
from slakh_dataset import SlakhDataset, INSTRUMENT_MAP

# In train_simple.py

# In train.py

# Replace your old function with this one
# In your training script (e.g., train.py)
class  LearningRateWarmup(tf.keras.callbacks.Callback):
    def __init__(self, warmup_steps, initial_lr,target_lr):
        super(LearningRateWarmup, self).__init__()
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.target_lr = target_lr
def on_train_batch_begin(self, batch, logs=None):
    if batch < self.warmup_steps:
        new_lr = self.initial_lr + (self.target_lr - self.initial_lr) * (batch / self.warmup_steps)

        lr = self.model.optimizer.learning_rate
        if hasattr(lr, "assign"):   # if it's a tf.Variable or tf.Tensor
            lr.assign(new_lr)
        else:                       # if it's a float
            self.model.optimizer.learning_rate = new_lr

        if batch % 50 == 0:
            current_lr = (lr.numpy() if hasattr(lr, "numpy") else lr)
            print(f"\nBatch {batch}: Learning rate is {current_lr:.7f}")



def build_model(input_shape, n_frames, n_pitches, n_instruments):
    """Builds a medium-sized, multi-task model."""
    
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_audio")
    
    # Spectrogram Layer
    x = tf.keras.layers.Lambda(lambda a: tf.abs(tf.signal.stft(a, frame_length=1024, frame_step=256)))(inputs)
    x = tf.keras.layers.Lambda(lambda s: tf.expand_dims(s, axis=-1))(x)
    
    # --- UPGRADED Shared Backbone ---
    # Block 1
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x) # Add Dropout for regularization

    # Block 2
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x) # Add Dropout for regularization
    
    # Block 3 (NEW) - This adds more power to learn complex features
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x) # Add Dropout for regularization

    backbone_output = x
    
    # --- Instrument Head (now more powerful) ---
    inst_head = tf.keras.layers.GlobalAveragePooling2D()(backbone_output)
    inst_head = tf.keras.layers.Dense(128, activation='relu')(inst_head) # Increased size
    instrument_output = tf.keras.layers.Dense(n_instruments, activation='softmax', name='instruments')(inst_head)
    
    # --- Transcription Head (connects to the new, deeper backbone) ---
    tx_head = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(backbone_output)
    tx_head = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(tx_head)
    tx_head = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(tx_head)
    tx_head_resized = tf.keras.layers.Resizing(height=n_frames, width=n_pitches)(tx_head)

    notes_output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='notes')(tx_head_resized)
    onsets_output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='onsets')(tx_head_resized)
    contours_output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='contours')(tx_head_resized)
    
    model_outputs = {"notes": notes_output, "onsets": onsets_output, "contours": contours_output, "instruments": instrument_output}
    return tf.keras.Model(inputs=inputs, outputs=model_outputs)

def train():
    # --- 1. Parameters ---
    BATCH_SIZE = 2
    EPOCHS = 50
    SAMPLE_RATE = 16000
    DURATION = 10.0
    HOP_LENGTH = 256
    TARGET_LEARNING_RATE = 5e-6
    
    # --- 2. Data Pipeline ---
    print("\n--- Building Data Pipeline ---")
    DATASET_PATH = 'C:/Users/Avi Pandey/Documents/research/pitnet/datasets/slakh2100_flac_redux' # UPDATE THIS PATH
    
    train_loader = SlakhDataset(DATASET_PATH, 'train', SAMPLE_RATE, DURATION, HOP_LENGTH)
    train_dataset = train_loader.create_tf_dataset(BATCH_SIZE)
    print(f"✓ Found {len(train_loader)} tracks for training.")
    
    val_loader = SlakhDataset(DATASET_PATH, 'validation', SAMPLE_RATE, DURATION, HOP_LENGTH)
    val_dataset = val_loader.create_tf_dataset(BATCH_SIZE, shuffle=False)
    print(f"✓ Found {len(val_loader)} tracks for validation.")
    
    # --- 3. Build and Compile Model ---
    print("\n--- Building and Compiling Model ---")
    input_samples = int(DURATION * SAMPLE_RATE)
    n_frames = input_samples // HOP_LENGTH
    
    model = build_model(
        input_shape=(input_samples,),
        n_frames=n_frames,
        n_pitches=88, 
        n_instruments=len(INSTRUMENT_MAP)
    )
    
    loss_weights = {
        "notes": 1.0,
        "onsets": 1.0,
        "contours": 1.0,
        "instruments": 1.0
    }
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=TARGET_LEARNING_RATE,
                                         beta_1=0.5,
                                         clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss={
            'notes': 'binary_crossentropy',
            'onsets': 'binary_crossentropy',
            'contours': 'binary_crossentropy',
            'instruments': 'categorical_crossentropy',
        },
        loss_weights=loss_weights, # <-- Add this argument
        metrics={
            'notes': 'accuracy',
            'onsets': 'accuracy',
            'contours': 'accuracy',
            'instruments': 'accuracy'
        }
    )
    model.summary()
    
# In your train() function

# --- 4. Set up Callbacks for Smarter Training ---
    print("\n--- Setting up callbacks ---")

# Define the learning rate for the main training phase
    

# Create the warm-up callback
    warmup_callback = LearningRateWarmup(
        warmup_steps=500,           # Warm up for the first 500 batches
        initial_lr=1e-8,            # Start with an extremely small LR
        target_lr=TARGET_LEARNING_RATE # Gradually increase to our target LR
    )

# Keep our other callbacks
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-8, verbose=1)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# --- 5. Train the Model ---
    print("\n--- Starting Training ---")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
    # Add the new warmup_callback to the list of callbacks
    callbacks=[warmup_callback, reduce_lr_callback, early_stopping_callback]
)
    print("\n--- Training Finished ---")

    print("\n--- Saving the best model ---")
    # Save the model's learned weights to a file
    model.save_weights("pit_net_model_final.weights.h5")
    print("✓ Model weights saved to pit_net_model_final.weights.h5")

    # In train_simple.py, after saving the model

    print("\n--- Evaluating on the test set ---")
    test_loader = SlakhDataset(
        data_dir=DATASET_PATH,
        split='test',
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        hop_length=HOP_LENGTH
    )
    test_dataset = test_loader.create_tf_dataset(batch_size=BATCH_SIZE, shuffle=False)
    
    print("Evaluating...")
    results = model.evaluate(test_dataset)
    print("\n--- Test Set Evaluation Results ---")
    # Keras returns a list of losses and metrics, this will help label them
    metric_names = ["total_loss"] + [f"{name}_loss" for name in model.output_names] + [f"{name}_accuracy" for name in model.output_names]
    for name, value in zip(metric_names, results):
        print(f"{name}: {value:.4f}")

if __name__ == '__main__':
    train() 