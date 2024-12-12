import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import cv2

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('train', target_size=(224, 224), batch_size=32, class_mode='categorical')
test_set = test_datagen.flow_from_directory('test', target_size=(224, 224), batch_size=32, class_mode='categorical')
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
predictions = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_set, epochs=10, validation_data=test_set)
test_loss, test_acc = model.evaluate(test_set)
print('Test accuracy:', test_acc)
img = cv2.imread('image2.jpg')
img = cv2.resize(img, (224, 224))
img = img.reshape((1, 224, 224, 3))
img = img.astype('float32') / 255.0

prediction = model.predict(img)
food_type = train_set.class_indices[prediction.argmax()]
print('Predicted food type:', food_type)
