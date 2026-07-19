import os

import seaborn as sns
from keras import models, layers
from matplotlib import pyplot as plt

os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from audioFiles import get_wav_files, getTrainWavFiles, getFileLabelTouple, TARGET_WORDS
import random
import numpy as np

label_names = np.array(TARGET_WORDS)

seed = 42
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

def get_audio_dataset(ds):
    paths, labels = zip(*ds)

    dataset = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    return dataset

def load_audio(path, label):
    audio = tf.io.read_file(path)
    audio, sample_rate = tf.audio.decode_wav(
        audio,
        desired_channels=1,
        desired_samples=16000
    )
    return audio, label

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)


if __name__ == '__main__':

    train_wav_files, validation_wav_files, test_wav_files = get_wav_files()

    yes_train_wav_files, no_train_wav_files, unknown_train_wav_files = (
        getTrainWavFiles(train_wav_files)
    )

    target = min(
        len(yes_train_wav_files),
        len(no_train_wav_files)
    )

    # unknown_train_wav_files = random.sample(
    #     unknown_train_wav_files,
    #     target*3
    # )
    #
    # train_wav_files = (
    #         yes_train_wav_files +
    #         no_train_wav_files +
    #         unknown_train_wav_files
    # )
    #
    train_ds, val_ds, test_ds = getFileLabelTouple(
        train_wav_files,
        validation_wav_files,
        test_wav_files
    )

    train_ds = get_audio_dataset(train_ds)
    val_ds = get_audio_dataset(val_ds)
    test_ds = get_audio_dataset(test_ds)

    train_ds = (
        train_ds
        .map(load_audio, num_parallel_calls=tf.data.AUTOTUNE)
        .map(squeeze, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(32)
    )

    val_ds = (
        val_ds
        .map(load_audio, num_parallel_calls=tf.data.AUTOTUNE)
        .map(squeeze, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(32)
    )

    test_ds = (
        test_ds
        .map(load_audio, num_parallel_calls=tf.data.AUTOTUNE)
        .map(squeeze, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(32)
    )




    train_spectrogram_ds = make_spec_ds(train_ds)
    val_spectrogram_ds = make_spec_ds(val_ds)
    test_spectrogram_ds = make_spec_ds(test_ds)

    for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
        break
    rows = 3
    cols = 3
    n = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

    for i in range(n):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        plot_spectrogram(example_spectrograms[i].numpy(), ax)
        ax.set_title(label_names[example_spect_labels[i].numpy()])

    plt.show()

    train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
    val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

    input_shape = example_spectrograms.shape[1:]
    print('Input shape:', input_shape)
    num_labels = len(label_names)

    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization()
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Downsample the input.
        layers.Resizing(32, 32),
        # Normalize.
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    EPOCHS = 20
    history = model.fit(
        train_spectrogram_ds,
        validation_data=val_spectrogram_ds,
        epochs=EPOCHS,
        class_weight={
            0: 1,
            1: 1,
            2: 0.2
        },
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )

    metrics = history.history
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('Loss [CrossEntropy]')

    plt.subplot(1, 2, 2)
    plt.plot(history.epoch, 100 * np.array(metrics['accuracy']), 100 * np.array(metrics['val_accuracy']))
    plt.legend(['accuracy', 'val_accuracy'])
    plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')

    results = model.evaluate(test_spectrogram_ds, return_dict=True)
    print(results)


    y_pred = model.predict(test_spectrogram_ds)
    y_pred = tf.argmax(y_pred, axis=1)
    y_true = tf.concat(list(test_spectrogram_ds.map(lambda s, lab: lab)), axis=0)
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,
                xticklabels=label_names,
                yticklabels=label_names,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()

    model.save("speech_command_model.keras")









