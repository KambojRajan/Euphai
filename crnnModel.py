import numpy as np
import os
import librosa
import parselmouth
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Flatten, Dense
from keras.optimizers import Adam
from preProcessing import segmentation, featureExtraction, featureNormalization, labeling


class CRNNTrainer:
    def __init__(self):
        pass

    @staticmethod
    def create_crnn_model(input_shape):
        crnn_branch = Sequential()
        crnn_branch.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
        crnn_branch.add(LSTM(128, return_sequences=True))
        crnn_branch.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        crnn_branch.add(LSTM(128, return_sequences=True))
        crnn_branch.add(Flatten())

        combined_model = Sequential()
        combined_model.add(crnn_branch)
        combined_model.add(Dense(64, activation='relu'))
        combined_model.add(Dense(1, activation='sigmoid'))

        return combined_model

    @staticmethod
    def train_and_save_crnn_model(input_data, output_model_file):
        try:
            segmented_audio, sr = segmentation(input_data)
            features = featureExtraction(segmented_audio, sr)
            features = featureNormalization(features)
            labeled_features = labeling(input_data, features)

            # Determine the input shape based on the features
            input_shape = (features[0].shape[1], features[0].shape[0)

            # Zero-pad the input audio segments to a desired length
            desired_length = 44100  # Adjust this to your desired length
            for i in range(len(segmented_audio)):
                if len(segmented_audio[i]) < desired_length:
                    zero_padding = np.zeros(desired_length - len(segmented_audio[i]))
                    segmented_audio[i] = np.concatenate((segmented_audio[i], zero_padding))

            # Create the CRNN model with the determined input shape
            model = CRNNTrainer.create_crnn_model(input_shape)

            # Compile the model
            optimizer = Adam(learning_rate=0.001)  # Adjust the learning rate as needed
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

            # Prepare the data for training
            X = np.array(features).transpose()
            speaker = labeled_features[3][0]
            y = np.array([speaker] * len(segmented_audio))

            # Train the CRNN model
            model.fit(X, y, epochs=1)

            # Save the trained model
            model.save(output_model_file)

        except Exception as e:
            print(f"Error processing file {input_data}: {str(e)}")
