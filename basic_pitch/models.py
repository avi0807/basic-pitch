#!/usr/bin/env python
# encoding: utf-8
#
# Copyright 2022 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Dict
import numpy as np
import tensorflow as tf

from basic_pitch import nn
from basic_pitch.constants import (
    ANNOTATIONS_BASE_FREQUENCY,
    ANNOTATIONS_N_SEMITONES,
    AUDIO_N_SAMPLES,
    AUDIO_SAMPLE_RATE,
    CONTOURS_BINS_PER_SEMITONE,
    FFT_HOP,
    N_FREQ_BINS_CONTOURS,
)
from basic_pitch.layers import signal, nnaudio

tfkl = tf.keras.layers

MAX_N_SEMITONES = int(np.floor(12.0 * np.log2(0.5 * AUDIO_SAMPLE_RATE / ANNOTATIONS_BASE_FREQUENCY)))

DEFAULT_LABEL_SMOOTHING = 0.2
DEFAULT_POSITIVE_WEIGHT = 0.5

CONTOUR_KERNEL_SIZE_1 = (5, 5)
CONTOUR_KERNEL_SIZE_2 = (3, 39)  # 3*13
CONTOUR_KERNEL_SIZE_3 = (5, 5)
CONTOUR_FILTERS_2 = 8

NOTES_KERNEL_SIZE_1 = (7, 7)
NOTES_STRIDES_1 = (1, 3)
NOTES_KERNEL_SIZE_2 = (7, 3)

ONSET_KERNEL_SIZE_1 = (5, 5)
ONSET_STRIDES_1 = (1, 3)
ONSET_KERNEL_SIZE_2 = (3, 3)


def transcription_loss(y_true: tf.Tensor, y_pred: tf.Tensor, label_smoothing: float) -> tf.Tensor:
    """Really a binary cross entropy loss. Used to calculate the loss between the predicted
    posteriorgrams and the ground truth matrices.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
        label_smoothing: Squeeze labels towards 0.5.

    Returns:
        The transcription loss.
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
    return bce


def weighted_transcription_loss(
    y_true: tf.Tensor, y_pred: tf.Tensor, label_smoothing: float, positive_weight: float = 0.5
) -> tf.Tensor:
    """The transcription loss where the positive and negative true labels are balanced by a weighting factor.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
        label_smoothing: Smoothing factor. Squeezes labels towards 0.5.
        positive_weight: Weighting factor for the positive labels.

    Returns:
        The weighted transcription loss.
    """
    negative_mask = tf.equal(y_true, 0)
    nonnegative_mask = tf.logical_not(negative_mask)
    bce_negative = tf.keras.losses.binary_crossentropy(
        tf.boolean_mask(y_true, negative_mask),
        tf.boolean_mask(y_pred, negative_mask),
        label_smoothing=label_smoothing,
    )
    bce_nonnegative = tf.keras.losses.binary_crossentropy(
        tf.boolean_mask(y_true, nonnegative_mask),
        tf.boolean_mask(y_pred, nonnegative_mask),
        label_smoothing=label_smoothing,
    )
    return ((1 - positive_weight) * bce_negative) + (positive_weight * bce_nonnegative)


def onset_loss(
    weighted: bool, label_smoothing: float, positive_weight: float
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """

    Args:
        weighted: Whether or not to use a weighted cross entropy loss.
        label_smoothing: Smoothing factor. Squeezes labels towards 0.5.
        positive_weight: Weighting factor for the positive labels.

    Returns:
        A function that calculates the transcription loss. The function will
        return weighted_transcription_loss if weighted is true else it will return
        transcription_loss.
    """
    if weighted:
        return lambda x, y: weighted_transcription_loss(
            x, y, label_smoothing=label_smoothing, positive_weight=positive_weight
        )
    return lambda x, y: transcription_loss(x, y, label_smoothing=label_smoothing)


def loss(
    label_smoothing: float = DEFAULT_LABEL_SMOOTHING,
    weighted: bool = False,
    positive_weight: float = DEFAULT_POSITIVE_WEIGHT,
) -> Dict[str, Any]:
    """Creates a keras-compatible dictionary of loss functions to calculate
    the loss for the contour, note and onset posteriorgrams.

    Args:
        label_smoothing: Smoothing factor. Squeezes labels towards 0.5.
        weighted: Whether or not to use a weighted cross entropy loss.
        positive_weight: Weighting factor for the positive labels.

    Returns:
        A dictionary with keys "contour," "note," and "onset" with functions as values to be used to calculate
        transcription losses.

    """
    loss_fn = lambda x, y: transcription_loss(x, y, label_smoothing=label_smoothing)
    loss_onset = onset_loss(weighted, label_smoothing, positive_weight)
    return {
        "contour": loss_fn,
        "note": loss_fn,
        "onset": loss_onset,
    }


def _initializer() -> tf.keras.initializers.VarianceScaling:
    return tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg", distribution="uniform", seed=None)


def _kernel_constraint() -> tf.keras.constraints.UnitNorm:
    return tf.keras.constraints.UnitNorm(axis=[0, 1, 2])


def get_cqt(inputs: tf.Tensor, n_harmonics: int, use_batchnorm: bool) -> tf.Tensor:
    """Calculate the CQT of the input audio.

    Input shape: (batch, number of audio samples, 1)
    Output shape: (batch, number of frequency bins, number of time frames)

    Args:
        inputs: The audio input.
        n_harmonics: The number of harmonics to capture above the maximum output frequency.
            Used to calculate the number of semitones for the CQT.
        use_batchnorm: If True, applies batch normalization after computing the CQT

    Returns:
        The log-normalized CQT of the input audio.
    """
    n_semitones = np.min(
        [
            int(np.ceil(12.0 * np.log2(n_harmonics)) + ANNOTATIONS_N_SEMITONES),
            MAX_N_SEMITONES,
        ]
    )
    x = nn.FlattenAudioCh()(inputs)
    x = nnaudio.CQT(
        sr=AUDIO_SAMPLE_RATE,
        hop_length=FFT_HOP,
        fmin=ANNOTATIONS_BASE_FREQUENCY,
        n_bins=n_semitones * CONTOURS_BINS_PER_SEMITONE,
        bins_per_octave=12 * CONTOURS_BINS_PER_SEMITONE,
    )(x)
    x = signal.NormalizedLog()(x)
    x = tf.keras.layers.Reshape(target_shape=x.shape[1:] + (1,))(x)
    if use_batchnorm:
        x = tfkl.BatchNormalization()(x)
    return x



def model(
    n_harmonics: int = 8,
    n_filters_contour: int = 32,
    n_filters_onsets: int = 32,
    n_filters_notes: int = 32,
    no_contours: bool = False,
) -> tf.keras.Model:
    """Basic Pitch's model implementation, updated for modern TensorFlow."""

    # --- Constants for model architecture ---
    CONTOUR_KERNEL_SIZE_1 = (5, 5)
    CONTOUR_KERNEL_SIZE_2 = (3, 39)
    CONTOUR_KERNEL_SIZE_3 = (5, 5)
    CONTOUR_FILTERS_2 = 8
    NOTES_KERNEL_SIZE_1 = (7, 7)
    NOTES_KERNEL_SIZE_2 = (7, 3)
    ONSET_KERNEL_SIZE_1 = (5, 5)
    ONSET_KERNEL_SIZE_2 = (3, 3)
    
    # --- Input and CQT Layer ---
    inputs = tf.keras.Input(shape=(None, 1), name="input_layer")
    x = get_cqt(inputs, n_harmonics, True)

    if n_harmonics > 1:
        x = nn.HarmonicStacking(
            CONTOURS_BINS_PER_SEMITONE, [0.5] + list(range(1, n_harmonics)), N_FREQ_BINS_CONTOURS
        )(x)
    else:
        x = nn.HarmonicStacking(CONTOURS_BINS_PER_SEMITONE, [1], N_FREQ_BINS_CONTOURS)(x)

    # --- Instrument Head (Your Custom Pathway) ---
    x_instrument = tfkl.Conv2D(32, (3, 3), padding="same", activation="relu", name="instrument_conv1")(x)
    x_instrument = tfkl.BatchNormalization(name="instrument_bn1")(x_instrument)
    x_instrument = tfkl.MaxPool2D(pool_size=(2, 2), name="instrument_pool1")(x_instrument)
    x_instrument = tfkl.Conv2D(64, (3, 3), padding="same", activation="relu", name="instrument_conv2")(x_instrument)
    x_instrument = tfkl.BatchNormalization(name="instrument_bn2")(x_instrument)
    x_instrument = tfkl.GlobalAveragePooling2D(name="instrument_gap")(x_instrument)
    x_instrument = tfkl.Dense(17, activation="softmax", name="instruments")(x_instrument)

    # --- Contour Pathway ---
    contours_name = "contours"
    x_contours = tfkl.Conv2D(n_filters_contour, CONTOUR_KERNEL_SIZE_1, padding="same", kernel_initializer=_initializer(), kernel_constraint=_kernel_constraint())(x)
    x_contours = tfkl.BatchNormalization()(x_contours)
    x_contours = tfkl.ReLU()(x_contours)
    x_contours = tfkl.Conv2D(CONTOUR_FILTERS_2, CONTOUR_KERNEL_SIZE_2, padding="same", kernel_initializer=_initializer(), kernel_constraint=_kernel_constraint())(x_contours)
    x_contours = tfkl.BatchNormalization()(x_contours)
    x_contours = tfkl.ReLU()(x_contours)
    x_contours_pre = tfkl.Conv2D(1, CONTOUR_KERNEL_SIZE_3, padding="same", activation="sigmoid", kernel_initializer=_initializer(), kernel_constraint=_kernel_constraint(), name="contours-reduced")(x_contours)
    x_contours = nn.FlattenFreqCh(name=contours_name)(x_contours_pre)
    x_contours_reduced = tfkl.Reshape(target_shape=x_contours.shape[1:] + (1,))(x_contours)

    # --- Note Pathway ---
    notes_name = "notes"
    x_notes = tfkl.Conv2D(n_filters_notes, NOTES_KERNEL_SIZE_1, padding="same", kernel_initializer=_initializer(), kernel_constraint=_kernel_constraint())(x_contours_reduced)
    x_notes = tfkl.ReLU()(x_notes)
    # Use AveragePooling to downsample the frequency dimension from 264 to 88
    x_notes = tfkl.AveragePooling2D(pool_size=(1, 3))(x_notes)
    x_notes_pre = tfkl.Conv2D(1, NOTES_KERNEL_SIZE_2, padding="same", kernel_initializer=_initializer(), kernel_constraint=_kernel_constraint(), activation="sigmoid")(x_notes)
    x_notes_pre = tfkl.Lambda(
        lambda x: tf.image.resize(x, (tf.shape(inputs)[1] // 512, 88))
    )(x_notes_pre)
    x_notes = nn.FlattenFreqCh(name=notes_name)(x_notes_pre)

    # --- Onset Pathway ---
    onsets_name = "onsets"
    x_onset = tfkl.Conv2D(n_filters_onsets, ONSET_KERNEL_SIZE_1, padding="same", kernel_initializer=_initializer(), kernel_constraint=_kernel_constraint())(x)
    x_onset = tfkl.BatchNormalization()(x_onset)
    x_onset = tfkl.ReLU()(x_onset)
    # Use AveragePooling to downsample the frequency dimension from 264 to 88
    x_onset = tfkl.AveragePooling2D(pool_size=(1, 3))(x_onset)
    x_onset = tfkl.Concatenate(axis=3, name="concat")([x_notes_pre, x_onset])
    x_onset = tfkl.Conv2D(1, ONSET_KERNEL_SIZE_2, padding="same", activation="sigmoid", kernel_initializer=_initializer(), kernel_constraint=_kernel_constraint())(x_onset)
    x_onset = tfkl.Lambda(
        lambda x: tf.image.resize(x, (tf.shape(inputs)[1] // 512, 88))
    )(x_onset)
    x_onset = nn.FlattenFreqCh(name=onsets_name)(x_onset)

    # --- Final Outputs ---
    outputs = {
        "notes": x_notes,
        "onsets": x_onset,
        "contours": x_contours,
        "instruments": x_instrument,
    }

    return tf.keras.Model(inputs=inputs, outputs=outputs)