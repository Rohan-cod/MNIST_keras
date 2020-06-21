from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.datasets.mnist import load_data
from keras.utils import to_categorical
import numpy as np
import os

# Train and Test Data
(train_data, train_labels), (test_data, test_labels) = load_data()

# Height, Width and number of channels
H = train_data.shape[1]
W = train_data.shape[2]
C = 1 # Grayscale

# Reshape to NHWC Format
train_data = np.reshape(train_data, (train_data.shape[0], H, W, C))
test_data = np.reshape(test_data, (test_data.shape[0], H, W, C))

# Scaling (range - (0.0, 1.0])
train_data = train_data.astype('float32') / 255.0
test_data = test_data.astype('float32') / 255.0

# One-Hot encoding for the Labels
num_classes = 10
train_labels_cat = to_categorical(train_labels, num_classes)
test_labels_cat = to_categorical(test_labels, num_classes)


# Shuffle the Training Data
for i in range(5):
	sh_index = np.random.permutation(len(train_data))

train_data = train_data[sh_index]
train_labels_cat = train_labels_cat[sh_index]

# Cross Validation set amount
cv_per = 0.1
cv_cnt = int(cv_per * len(train_data))

# CV set
cv_data = train_data[:cv_cnt, :]
cv_labels_cat = train_labels_cat[:cv_cnt, :]

# Remaining Training set
train_data_rem = train_data[cv_cnt:, :]
train_labels_cat_rem = train_labels_cat[cv_cnt:, :]


# Model
def cnn_model():
	model = Sequential()

	# Convolutional and MaxPooling Layers
	model.add(Conv2D(filters=32,
		kernel_size=(3,3),
		activation='relu',
		padding='same',
		input_shape=(H, W, C)
	))

	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(filters=64,
		kernel_size=(3,3),
		activation='relu',
		padding='same'
	))

	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(filters=64,
		kernel_size=(3,3),
		activation='relu',
		padding='same',
	))

	model.add(MaxPooling2D(pool_size=(2,2)))

	# Flatten Input Layer
	model.add(Flatten())

	# Dense Layer
	model.add(Dense(128,
		activation='relu'
	))

	# Output Layer
	model.add(Dense(num_classes,
		activation='softmax'
	))

	# Compile Model
	model.compile(optimizer='adam',
		loss='categorical_crossentropy',
		metrics=['accuracy']
	)

	# Return Model
	return model


# Initialize the Model
model = cnn_model()

# Train the Model
result = model.fit(train_data_rem,
	train_labels_cat_rem,
	epochs=15,
	batch_size=64,
	validation_data=(cv_data, cv_labels_cat)
)

# Test the Model
test_loss, test_accuracy = model.evaluate(test_data,
	test_labels_cat,
	batch_size=64
)

# Print Test Loss and Test Accuracy
print('Test Loss: %.4f & Test Accuracy: %.4f' % (test_loss, test_accuracy))

# Serialize Model to JSON
model_json = model.to_json()

# Save Model to File
with open("model.json", 'w') as json_file:
	json_file.write(model_json)

# Save Model Weights
model.save_weights("model.h5")

print("Model Saved")