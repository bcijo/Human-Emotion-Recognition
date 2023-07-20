import speech_recognition
import wave
import tensorflow as tf
from transformers import TFRobertaForSequenceClassification
from transformers import RobertaTokenizerFast
import numpy as np
import librosa

dir_path = 'saved_text_model-RoBERTa_based'
tokenizer_fine_tuned = RobertaTokenizerFast.from_pretrained(dir_path)
model_fine_tuned = TFRobertaForSequenceClassification.from_pretrained(dir_path)
audio_model = tf.keras.models.load_model('saved_audio_model-LSTM_based')


def recognize_speech():
    recognizer = speech_recognition.Recognizer()

    while True:
        try:
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration=0.5)
                print("Listening...")
                audio = recognizer.listen(mic)
                # print(type(audio))

                text = recognizer.recognize_google(audio)
                text = text.lower()
                print('Recognized: ', text)

                if text == 'stop':
                    break

                # Save the audio as a .wav file
                with wave.open("audio1.wav", "wb") as file:
                    file.setnchannels(1)  # stereo recording
                    file.setsampwidth(4)  # higher quality audio
                    file.setframerate(22400)  # higher sampling rate
                    file.writeframes(audio.get_wav_data())

                text_preds = text_emotion_prediction(text)
                audio_preds = audio_emotion_prediction('audio1.wav')
                ensemble(text_preds, audio_preds)

        except speech_recognition.UnknownValueError:
            recognizer = speech_recognition.Recognizer()
            continue


def get_audio_features(audio):
    y, sr = librosa.load(audio, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15, n_fft=2048, hop_length=512).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    audio_features = np.hstack((mfcc, zcr, rms, mel))
    audio_features = np.expand_dims(audio_features, -1)
    audio_features = np.expand_dims(audio_features, 0)
    return audio_features


def text_emotion_prediction(text):
    resp_token = tokenizer_fine_tuned(text, truncation=True, padding=True, return_tensors='tf')
    new_predictions = model_fine_tuned.predict(dict(resp_token))
    vec = new_predictions.logits[0]
    after_softmax = np.exp(vec) / sum(np.exp(vec))
    after_softmax.reshape((1, -1))
    pred = np.argmax([after_softmax], axis=1)
    after_neutral = after_softmax.tolist()
    after_neutral.append(1 - after_softmax[pred[0]])
    return np.array(after_neutral)


def audio_emotion_prediction(audio):
    features = get_audio_features(audio)
    audio_preds = audio_model.predict([features])
    return np.array(audio_preds)


def ensemble(text_preds, audio_preds):
    emotion_set = ['sad', 'joy', 'anger', 'fear','surprise', 'neutral']
    text_weight = 0.6
    audio_weight = 0.4

    text_pred = np.argmax([text_preds], axis=1)
    audio_pred = np.argmax(audio_preds, axis=1)

    final_preds = text_preds * text_weight + audio_preds * audio_weight
    final_pred = np.argmax(final_preds, axis=1)

    print('\nText model prediction : ', emotion_set[text_pred[0]])
    print('Text model accuracy for prediction : ', text_preds[text_pred[0]])

    print('\nAudio model prediction : ', emotion_set[audio_pred[0]])
    print('Audio model accuracy for prediction : ', audio_preds[0][audio_pred[0]])

    print('\n\nFinal prediction : ', emotion_set[final_pred[0]])
    print('Final accuracy for prediction : ', final_preds[0][final_pred[0]])


recognize_speech()