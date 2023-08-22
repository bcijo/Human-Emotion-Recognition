# Human-Emotion-Recognition

## Introduction

- Human Emotion is a complex psychological experience that can be triggered by various internal and external factors.
- Emotion recognition is the idea of identifying and understanding other people’s emotions based on verbal and non-verbal   cues like, voice, speech patterns, facial expressions and body language. Understanding emotions helps us to respond appropriately to other people’s  needs.

## About the Project 

- In here we focus on enhancing the human speech emotion using lexical features.
- Lexical features refer to the words and language used by an individual, while acoustic features refer to the characteristics of their voice, such as pitch, tone, and volume.
- By analyzing both lexical and acoustic features, we can gain a more comprehensive understanding of an individual's emotional state

## Datasets used for the project 

#### Audio Dataset : RAVEDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
RAVEDESS was the audio dataset we worked on . It consists of 8 emotion classes namely Sad, Anger, Happy, Surprised , Disgust, Neutral and Calm with the total size of 1440. But the dataset was modified to match the dataset of the lexical model which had only 6 classes. So after removal of disgust class and merging calm with Neutral the final size of the dataset was 1056.


### Text Dataset : dair-ai/emotion from HuggingFace
The text dataset is taken from dair-ai/emotion from hugging face. It is a very clean dataset consisting of 6 basic emotions - sad, happy, anger, love, surprise and fear. But in order to have the same classes as the Audio dataset, ‘love’ class was removed. The dataset consists of 417,000 datapoints belonging to different classes which is mostly balanced.

## Modelling and Results
- The audio model was built by extracting the audio features using Librosa library and passing the extracted featrues through Bi-LSTM model, the audio model achieved an accuracy upto 64% on testing data from RAVEDESS.
- The text model was bulit by passing the cleaned text to RoBERTa Tokenizer and then through a fine-tuned RobertSequenceClassification model, this fine-tuned text model achieved an accuracy of close to 96% on dair-ai/emotion dataset.
- The final ensemble model was then built using weighted means method to combine the two models to give a combined output.

### Research Papers referred : 
- [A Novel Multi-Window Spectrogram Augmentation
Approach for Speech Emotion Recognition using
Deep Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9692411)
- [LSTM-based Text Emotion Recognition Using
Semantic and Emotional Word Vectors](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8470378)
- [Emotion Recognition using Acoustic and Lexical Features](https://www.isca-speech.org/archive_v0/archive_papers/interspeech_2012/i12_0366.pdf)
- [Emotion Recognition Combining Acoustic and
Linguistic Features Based on Speech Recognition
Results](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9621810)


