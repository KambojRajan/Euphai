from crnnModel import train_and_save_crnn_model
import os

def train_models_for_directory(root_directory, output_directory):
    file_paths = []

    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_paths.append(file_path)

    for file_path in file_paths:
        output_model_file = os.path.join(output_directory, f"{os.path.splitext(os.path.basename(file_path))[0]}_model.h5")
        train_and_save_crnn_model(file_path, output_model_file)

if __name__ == "__main__":
    input_directory = "/content/dataset/50_speakers_audio_data"
    output_directory = "/content/drive/MyDrive/Audio"

    train_models_for_directory(input_directory, output_directory)
