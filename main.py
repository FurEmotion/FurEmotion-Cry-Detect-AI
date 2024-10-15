# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio

yamnet_base = os.getcwd()
sys.path.append(yamnet_base)

import params as yamnet_params
import yamnet_new as yamnet_model
import features as features_lib

chkp = True

params = yamnet_params.Params()

# 사용자 정의 데이터셋을 위한 csv 파일 생성
data = 'furmotion_class_map.csv'

# 데이터를 로드합니다.
df = pd.read_csv(data)

# 클래스 이름 정의
class_names = yamnet_model.class_names(data)

# YAMNet 모델 정의. yamnet_frames_model_transfer 함수를 수정하여 사용자 정의 데이터셋에 사용함.


@tf.function
def load_wav_16k_mono(filename):
    """ WAV 파일을 로드하고, float 텐서로 변환하며, 16kHz 단일 채널 오디오로 리샘플링. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
        file_contents,
        desired_channels=1)

    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)

    return wav


def load_wav_for_map(filename, label, fold):
    return load_wav_16k_mono(filename), label, fold


# 학습 데이터 열기(8:1:1로 분할함)
data1 = 'furmotion_class_map_train.csv'
df_train = pd.read_csv(data1)

df_train.drop(['category'], inplace=True, axis=1)

# folds는 데이터셋을 학습, 검증 및 테스트로 나누기 위해 사용됨.
filenames = df_train['filenames']
targets = df_train['targets']
folds = df_train['folds']

# 이 함수는 WAV 파일을 받고 각 WAV 파일을 96ms 길이의 프레임으로 나누며, 10ms 홉을 사용.
# 각 프레임의 레이블은 해당 오디오 파일의 레이블입니다. 이후 이러한 프레임의 배치를 YAMNet 모델에 입력으로 사용함.


def yamnet_frames_model_transfer1(wav_data, label, fold):

    waveform_padded = features_lib.pad_waveform(wav_data, params)
    log_mel_spectrogram, features = features_lib.waveform_to_log_mel_spectrogram_patches(
        waveform_padded, params)
    num_embeddings = tf.shape(features)[0]
    print(log_mel_spectrogram.shape)

    return log_mel_spectrogram, label, fold


main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))

# load_wav_for_map 함수를 데이터셋에 적용. 아직 메모리에 로드된 것은 아님.
main_ds = main_ds.map(load_wav_for_map)
main_ds.element_spec

# WAV 파일을 프레임으로 나.
# 주어진 WAV 파일에 대해 yamnet_frames_model_transfer1 함수의 출력 크기는 (m, 96, 64)이고
# unbatch()를 사용하면 (96, 64) 크기의 m개의 텐서를 얻을 수 있음.
# 결론적으로 이러한 배열 중 (96, 64) 크기의 배치가 모델의 입력으로 사용됨.
main_ds = main_ds.map(yamnet_frames_model_transfer1)

# 마지막 단계로 학습 중에 사용하지 않을 fold 열을 데이터셋에서 제거함.
cached_ds = main_ds.cache()
train_ds = cached_ds.filter(lambda embedding, label, folds: folds <= 3)
val_ds = cached_ds.filter(lambda embedding, label, folds: folds <= 4)
test_ds = cached_ds.filter(lambda embedding, label, fold: fold == 5)

# 더 이상 필요하지 않은 fold 열을 제거.


def remove_fold_column(embedding, label, fold): return (embedding, label)


train_ds = train_ds.map(remove_fold_column)
val_ds = val_ds.map(remove_fold_column)
test_ds = test_ds.map(remove_fold_column)

# 크기 (96, 64)의 프레임들로 이루어진 배치를 생성. 배치 크기는 32.
# 학습 데이터셋을 셔플하여 동일한 오디오의 프레임들이 한 배치에 포함되지 않도록 합니다.
train_ds = train_ds.cache().shuffle(1000).batch(
    32 * 2).prefetch(tf.data.experimental.AUTOTUNE)
val_ds = val_ds.cache().batch(32 * 2).prefetch(tf.data.experimental.AUTOTUNE)

# YAMNet 모델을 로드합니다. yamnet_frames_model_transfer1은 yamnet.py 파일의 yamnet_frames_model_transfer의 수정 버전입니다.
# YAMNet을 처음부터 학습할 수 있도록 코드를 조금 수정.

yamnet = yamnet_model.yamnet_frames_model_transfer(params, 1)

preloaded_layers = yamnet.layers.copy()
preloaded_weights = []
for pre in preloaded_layers:
    preloaded_weights.append(pre.get_weights())

# 마지막 레이어를 제외하고, 사전 학습된 모델에서 가중치를 로드.
# 사전 학습된 가중치를 로드한 레이어와 그렇지 않은 레이어를 확인.
if chkp == True:
    yamnet.load_weights(os.path.join(yamnet_base, 'yamnet.h5'), by_name=True)
    for layer, pre in zip(yamnet.layers, preloaded_weights):
        weights = layer.get_weights()
        if weights:
            if np.array_equal(weights, pre):
                print('not loaded', layer.name)
            else:
                print('loaded', layer.name)

NAME = 'yamnet.h5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    NAME, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            restore_best_weights=True)

if os.path.exists('logs'):
    os.makedirs('logs', exist_ok=True)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs')

yamnet.compile(optimizer='adam',
               loss="BinaryCrossentropy",
               metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

yamnet.fit(train_ds, epochs=100, validation_data=val_ds,
           callbacks=[checkpoint, tensorboard, callback])

# 테스트 데이터셋을 사용하여 모델 평가
eval_loss = yamnet.evaluate(test_ds)
print('loss=', eval_loss)

# 모델 테스트
import os
dir_ = "test_data.csv"
df_test_b = pd.read_csv(dir_)
base_data_path = 'data/wav/test'
full_path = df_test_b['filename'].apply(
    lambda row: os.path.join(base_data_path, row))
df_test_b = df_test_b.assign(filename=full_path)

full_path = df_test_b['filename'].apply(lambda row: (row + '.wav'))

df_test_b = df_test_b.assign(filename=full_path)

filenames = df_test_b['filename']
targets = df_test_b['target']
df_test_b['fold'] = 1
folds = df_test_b['fold']

# .wav 파일이 포함된 디렉토리
test_b = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))
test_b = test_b.map(load_wav_for_map)
test_b = test_b.map(yamnet_frames_model_transfer1).unbatch()
def remove_fold_column(embedding, label, fold): return (embedding, label)


test_b = test_b.map(remove_fold_column)
test_b = test_b.cache().batch(32).prefetch(tf.data.experimental.AUTOTUNE)

evaluate = yamnet.evaluate(test_b)
