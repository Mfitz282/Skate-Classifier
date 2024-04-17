import os
import numpy as np
from keras.models import load_model

# Load the model
model_path = 'binary_classification_model.h5'
model = load_model(model_path)

# Directory containing .npy files
directory = '/Users/mfitzpatrick/Pictures/GoPro/SkateModelClips/Testing Data/2022-09-24 Skate Romsey/HERO8 Black 1/avi/nparrays'

# Output file path
output_file_path = 'predictions.txt'

# Iterate over .npy files in the directory
with open(output_file_path, 'w') as output_file:
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory, filename)
            data = np.load(file_path)
            data = np.expand_dims(data, axis=0)
            data = data.reshape(-1, 20, 8)

            # Perform prediction
            predictions = model.predict(data)
            if predictions >= 0.5:
                classification = 'Make'
            else:
                classification = 'Bail'

            # Write predictions to the output file
            output_file.write(f'{filename}, {predictions}, {classification}\n')


print('Predictions saved to', output_file_path)