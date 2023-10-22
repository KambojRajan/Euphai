import numpy as np
import librosa
import parselmouth
from sklearn.preprocessing import StandardScaler
from keras.layers import Conv1D, LSTM, Flatten, Dense
from keras.models import Sequential


class FeatureExtractor:
    def __init__(self):
        pass

    @staticmethod
    def segmentation(file, segment_length=2, overlap=0.25):
        try:
            audio, sr = librosa.load(file, sr=None)
            start = 0
            step = int(segment_length * sr * (1 - overlap))
            segmented_audio = []
            while start + sr * segment_length < len(audio):
                segment = audio[start:start + sr * segment_length]
                segmented_audio.append(segment)
                start += step
            return np.array(segmented_audio), sr
        except Exception as e:
            print(f"Error in segmentation for file {file}: {str(e)}")
            return None, None

    @staticmethod
    def cal_mfcc(segmented_audio, sr):
        try:
            mfccs_list = []
            for segment in segmented_audio:
                mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
                mfccs = np.expand_dims(mfccs, axis=0)
                mfccs_list.extend(mfccs)
            mfccs_2d = np.concatenate(mfccs_list, axis=1)
            mfccs_1d = mfccs_2d.flatten()
            return mfccs_1d
        except Exception as e:
            print(f"Error in MFCC calculation: {str(e)}")
            return np.array([])

    @staticmethod
    def cal_pitch(segmented_audio, sr):
        try:
            pitches = []
            for audio in segmented_audio:
                audio_float64 = audio.astype(np.float64)
                sound = parselmouth.Sound(values=audio_float64, sampling_frequency=sr)
                pitch = sound.to_pitch()
                if pitch is not None:
                    pitch_values = pitch.selected_array['frequency']
                    mean_pitch = np.mean(pitch_values)
                    pitches.append(np.array(mean_pitch))
            return np.array(pitches)
        except Exception as e:
            print(f"Error in pitch calculation: {str(e)}")
            return np.array([])

    @staticmethod
    def cal_formants(segmented_audio, sr):
        try:
            all_formants = []
            for audio in segmented_audio:
                audio_float64 = audio.astype(np.float64)
                sound = parselmouth.Sound(audio_float64, sampling_frequency=sr)
                formants = sound.to_formant_burg(time_step=0.010, maximum_formant=5000)
                if formants is not None:
                    formant_values = []
                    for t in formants.ts():
                        f1 = formants.get_value_at_time(formant_number=1, time=t)
                        f2 = formants.get_value_at_time(formant_number=2, time=t)
                        f3 = formants.get_value_at_time(formant_number=3, time=t)
                        f4 = formants.get_value_at_time(formant_number=4, time=t)
                        f5 = formants.get_value_at_time(formant_number=5, time=t)
                        if np.isnan(f5):
                            f5 = 0
                        formant_values.extend(np.array([f1, f2, f3, f4, f5]))
                    all_formants.extend(np.array(formant_values))
            return np.array(all_formants)
        except Exception as e:
            print(f"Error in formant calculation: {str(e)}")
            return np.array([])

    @staticmethod
    def feature_extraction(segmented_audio, sr):
        mfccs = FeatureExtractor.cal_mfcc(segmented_audio, sr)
        pitch = FeatureExtractor.cal_pitch(segmented_audio, sr)
        formants = FeatureExtractor.cal_formants(segmented_audio, sr)
        valid_features = [mfccs, pitch, formants]
        return valid_features

    @staticmethod
    def feature_normalization(features):
        scalers = [StandardScaler() for _ in features]
        normalized_features = []
        for i, feature in enumerate(features):
            num_coefficients = feature.shape[0]
            reshaped_feature = feature.reshape(num_coefficients, -1)
            normalized_feature = scalers[i].fit_transform(reshaped_feature)
            normalized_features.append(normalized_feature)
        return normalized_features


class Preprocessing:
    def __init__(self):
        pass

    @staticmethod
    def labeling(path, normalized_features):
        try:
            path_components = path.split('/')
            speaker_directory = path_components[-2]
            speaker_number = int(speaker_directory[-4:])
            if not isinstance(normalized_features, list):
                normalized_features = [normalized_features]
            normalized_features.append([speaker_number])
            return normalized_features
        except Exception as e:
            print(f"Error in labeling: {str(e)}")
            return []
