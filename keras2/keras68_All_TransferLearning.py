from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from tensorflow.keras.applications import Xception

model_list = [VGG16(), VGG19(), ResNet50(), ResNet50V2(), ResNet101(), ResNet101V2(), ResNet152(), ResNet152V2(), DenseNet121(), DenseNet169(), DenseNet201(), InceptionV3(), InceptionResNetV2(), MobileNet(), MobileNetV2(),
MobileNetV3Small(), MobileNetV3Large(), NASNetLarge(), NASNetMobile(), EfficientNetB0(), EfficientNetB1(), EfficientNetB7(), Xception()]

for model in model_list:
    model = model
    print(f"모델명 : {model.name}")
    print(f"전체 가중치 갯수   : , {len(model.weights)}")
    print(f"훈련 가능 가중치 갯수   : ,{len(model.trainable_weights)}")        

'''
# model = VGG16()
# model = VGG19()
# model = ResNet50()
# model = ResNet50V2()
# model = ResNet101()
# model = ResNet101V2()
# model = ResNet152()
# model = ResNet152V2()
# model = DenseNet121()
# model = DenseNet169()
# model = DenseNet201()
# model = InceptionV3()
# model = InceptionResNetV2()
# model = MobileNet()
# model = MobileNetV2()
# model = MobileNetV3Small()
# model = MobileNetV3Large()
# model = NASNetLarge()
# model = NASNetMobile()
# model = EfficientNetB0()
# model = EfficientNetB1()
# model = EfficientNetB7()
model = Xception()

# model.trainable = False
model.summary()

print("=======================================================")
print("모델명 : ", Xception)
print("전체 가중치 갯수   : ", len(model.weights))
print("훈련 가능 가중치 갯수   : ", len(model.trainable_weights))
'''

# =======================================================
# 모델명 :  <function VGG16 at 0x000002E337E78A60>
# 전체 가중치 갯수   :  32
# 훈련 가능 가중치 갯수   :  32
# =======================================================
# 모델명 :  <function VGG19 at 0x00000240F76251F0>
# 전체 가중치 갯수   :  38
# 훈련 가능 가중치 갯수   :  38
# =======================================================
# 모델명 :  <function ResNet50 at 0x000001C524AE8670>
# 전체 가중치 갯수   :  320
# 훈련 가능 가중치 갯수   :  214
# =======================================================
# 모델명 :  <function ResNet50V2 at 0x000001D2F9DA7AF0>
# 전체 가중치 갯수   :  272
# 훈련 가능 가중치 갯수   :  174
# =======================================================
# 모델명 :  <function ResNet101 at 0x0000020B17CB7670>
# 전체 가중치 갯수   :  626
# 훈련 가능 가중치 갯수   :  418
# =======================================================
# 모델명 :  <function ResNet101V2 at 0x000001AC76D57C10>
# 전체 가중치 갯수   :  544
# 훈련 가능 가중치 갯수   :  344
# =======================================================
# 모델명 :  <function ResNet152 at 0x0000014A2B708790>
# 전체 가중치 갯수   :  932
# 훈련 가능 가중치 갯수   :  622
# =======================================================
# 모델명 :  <function ResNet152V2 at 0x000001F6F8B99D30>
# 전체 가중치 갯수   :  816
# 훈련 가능 가중치 갯수   :  514
# =======================================================
# 모델명 :  <function DenseNet121 at 0x000002971FD05DC0>
# 전체 가중치 갯수   :  606
# 훈련 가능 가중치 갯수   :  364
# =======================================================
# 모델명 :  <function DenseNet169 at 0x000001E95AC16E50>
# 전체 가중치 갯수   :  846
# 훈련 가능 가중치 갯수   :  508
# =======================================================
# 모델명 :  <function DenseNet201 at 0x000001D35F655F70>
# 전체 가중치 갯수   :  1006
# 훈련 가능 가중치 갯수   :  604
# =======================================================
# 모델명 :  <function InceptionV3 at 0x000001C4081AE940>
# 전체 가중치 갯수   :  378
# 훈련 가능 가중치 갯수   :  190
# =======================================================
# 모델명 :  <function InceptionResNetV2 at 0x000002BF0F6FECA0>
# 전체 가중치 갯수   :  898
# 훈련 가능 가중치 갯수   :  490
# =======================================================
# 모델명 :  <function MobileNet at 0x00000177D2BC13A0>
# 전체 가중치 갯수   :  137
# 훈련 가능 가중치 갯수   :  83
# =======================================================
# 모델명 :  <function MobileNetV2 at 0x00000238B56B2790>
# 전체 가중치 갯수   :  262
# 훈련 가능 가중치 갯수   :  158
# =======================================================
# 모델명 :  <function MobileNetV3Small at 0x00000226B1662E50>
# 전체 가중치 갯수   :  210
# 훈련 가능 가중치 갯수   :  142
# =======================================================
# 모델명 :  <function MobileNetV3Large at 0x0000012D2D6D3F70>
# 전체 가중치 갯수   :  266
# 훈련 가능 가중치 갯수   :  174
# =======================================================
# 모델명 :  <function NASNetLarge at 0x000001F7B5A32C10>
# 전체 가중치 갯수   :  1546
# 훈련 가능 가중치 갯수   :  1018
# =======================================================
# 모델명 :  <function NASNetMobile at 0x00000247A01E1C10>
# 전체 가중치 갯수   :  1126
# 훈련 가능 가중치 갯수   :  742
# =======================================================
# 모델명 :  <function EfficientNetB0 at 0x000001D2328FE310>
# 전체 가중치 갯수   :  314
# 훈련 가능 가중치 갯수   :  213
# =======================================================
# 모델명 :  <function EfficientNetB1 at 0x0000020EBAE8E3A0>
# 전체 가중치 갯수   :  442
# 훈련 가능 가중치 갯수   :  301
# =======================================================
# 모델명 :  <function EfficientNetB7 at 0x00000270DBF1E700>
# 전체 가중치 갯수   :  1040
# 훈련 가능 가중치 갯수   :  711
# =======================================================
# 모델명 :  <function Xception at 0x0000013E956884C0>
# 전체 가중치 갯수   :  236
# 훈련 가능 가중치 갯수   :  156
