from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

# model.trainable = False

# for layer in model.layers:
#     layer.trainable = False

model.layers[0].trainable = False   # Dense
# model.layers[1].trainable = False   # Dense_1
# model.layers[2].trainable = False   # Dense_2

model.summary()

print(model.layers)