import streamlit as st
# import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import speech_recognition as sr
import wave
import transformers
from transformers import TFRobertaForSequenceClassification
from transformers import RobertaTokenizerFast
import csv

audio_model = tf.keras.models.load_model("saved_audio_model-20230523T172522Z-001/saved_audio_model")
#dir_path=r"C:\Users\sakshi\Downloads\roberta-20230521T191714Z-001\roberta"
tokenizer_fine_tuned = RobertaTokenizerFast.from_pretrained('bcijo/Emotion-RoBERTa')
model_fine_tuned = TFRobertaForSequenceClassification.from_pretrained('bcijo/Emotion-RoBERTa')

def preprocess(audio_path):
    y, sr = librosa.load(audio_path, duration=3 , offset= 0.5)
    audio, _ = librosa.effects.trim(y)
    audio = librosa.util.fix_length(audio, size=66150)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15,n_fft=2048, hop_length=512).T, axis=0)
    print(mfcc.shape)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
    print(zcr.shape)
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
    print(rms.shape)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    features = np.hstack((mfcc, zcr, rms, mel))
    features = np.expand_dims(features, -1)
    features = np.expand_dims(features, 0)
    print(features.shape)
    return features

def text_emotion_prediction(text):
    resp_token = tokenizer_fine_tuned(text, truncation=True, padding=True, return_tensors='tf')
    new_predictions = model_fine_tuned.predict(dict(resp_token))
    vec = new_predictions.logits[0]
    after_softmax = np.exp(vec) / sum(np.exp(vec))
    after_softmax.reshape((1, -1))
    pred = np.argmax([after_softmax], axis=1)
    after_neutral = after_softmax.tolist()
    after_neutral.append(1 - after_softmax[pred[0]])
    return after_neutral
def audio_emotion_prediction(audio_features):
    audio_preds = audio_model.predict(audio_features)
    return audio_preds

def save_to_csv(data, label):
    with open('user_inputs.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([data, label])

def ensemble(text_preds, audio_preds):
    emotion_set = ['sad', 'joy', 'anger', 'fear','surprise', 'neutral']
    text_weight = np.array([0.6], dtype=np.float32)
    audio_weight = np.array([0.4], dtype=np.float32)

    text_pred = np.argmax([text_preds], axis=1)
    audio_pred = np.argmax(audio_preds, axis=1)

    final_preds = text_preds * text_weight + audio_preds * audio_weight
    final_pred = np.argmax(final_preds, axis=1)

    print('\nText model prediction : ', emotion_set[text_pred[0]])
    st.write(f'Text model prediction: {emotion_set[text_pred[0]]}')
    print('Text model accuracy for prediction : ', text_preds[text_pred[0]])
    st.write(f'Text model accuracy for prediction: {text_preds[text_pred[0]]}')

    print('\nAudio model prediction : ', emotion_set[audio_pred[0]])
    st.write(f'Audio model prediction: {emotion_set[audio_pred[0]]}')
    print('Audio model accuracy for prediction : ', audio_preds[0][audio_pred[0]])
    st.write(f'Audio model accuracy for prediction: {audio_preds[0][audio_pred[0]]}')

    print('\n\nFinal prediction : ', emotion_set[final_pred[0]])
    st.write(f'Final prediction: {emotion_set[final_pred[0]]}')
    print('Final accuracy for prediction : ', final_preds[0][final_pred[0]])
    st.write(f'Final accuracy for prediction: {final_preds[0][final_pred[0]]}')

# Define the Streamlit app
def app():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Select an option",
                                    ["Home", "Prediction"])

    if app_mode == "Home":
        st.title("Welcome to Emotion Recognition App")
        # st.write("This is the home page of the app.")
        st.image("dsbck.jpg", width=900)
    elif app_mode == "Prediction":
        st.title("Real-time Emotion Recognition")

        # Define the duration of recording in seconds
        duration = 5

        # Create a button to start recording
        if st.button("Start Recording"):
            r = sr.Recognizer()
            with sr.Microphone() as source:
                st.write("Recording started...")
                audio = r.listen(source)
                st.write("Recording completed.")

                try:
                    text = r.recognize_google(audio)
                    st.write("Text: ", text)
                except sr.UnknownValueError:
                    st.write("Speech recognition could not understand audio.")
                except sr.RequestError as e:
                    st.write("Error occurred during speech recognition:", str(e))

            with wave.open("audio1.wav", "wb") as file:
                file.setnchannels(1)  # stereo recording
                file.setsampwidth(4)  # higher quality audio
                file.setframerate(22050)  # higher sampling rate
                file.writeframes(audio.get_wav_data())

            audio = preprocess('audio1.wav')
            print(audio)

            final_audio = audio_emotion_prediction(audio)
            # st.write(f"final_audio:{final_audio}")


            final_text = text_emotion_prediction(text)

            ensemble(final_text,final_audio)

def callback(indata, outdata, frames, time, status):
    # This is a simple audio callback that does nothing.
    pass


if __name__ == "__main__":
    app()

with sd.Stream(callback=callback):
    print("Sounddevice is working correctly.")
    input("Press Enter to exit.")
