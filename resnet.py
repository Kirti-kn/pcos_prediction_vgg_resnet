from keras.applications import ResNet101
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

image_size = (100, 100)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1
)

test_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.2
)

train_generator = train_datagen.flow_from_directory(
    'train_directory2',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'train_directory2',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Load pre-trained ResNet101 model without top layers
resnet_base = ResNet101(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

model = Sequential()

# Add ResNet101 base model
model.add(resnet_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.summary()

# Freeze ResNet101 base layers
resnet_base.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, epochs=20, validation_data=test_generator)

model.save('pcos_model_resnet101.h5')
