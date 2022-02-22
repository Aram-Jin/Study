from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img_path = '../_data/cat_dog.jpeg'
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
print("==================================== image.img_to_array(img) ===================================")
print(x, '\n', x.shape)    # (224, 224, 3)

x = np.expand_dims(x, axis=0)
print("==================================== np.expand_dims(x, axis=0) ===================================")
print(x, '\n', x.shape)    # (1, 224, 224, 3)

x = preprocess_input(x)
print("==================================== preprocess_input(x) ===================================")
print(x, '\n', x.shape)    # (1, 224, 224, 3)

preds = model.predict(x)
print(preds, '\n', preds.shape)    #  (1, 1000)

print('결과는 :', decode_predictions(preds, top=5)[0])
# 결과는 : [('n02093428', 'American_Staffordshire_terrier', 0.47699827), ('n02108089', 'boxer', 0.436985), ('n02087394', 'Rhodesian_ridgeback', 
# 0.02643497), ('n02107574', 'Greater_Swiss_Mountain_dog', 0.015008465), ('n02093256', 'Staffordshire_bullterrier', 0.010407576)]