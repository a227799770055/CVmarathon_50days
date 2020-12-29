# CVmarathon_50days
### Computer Vision and Deep Learning Marathson




- [x] Day01 OpenCV 簡介與開啟圖片 [作業連結](https://github.com/a227799770055/CVmarathon_50days/blob/main/D1/Day01.ipynb)
- [x] Day02 Color Presentation [作業連結](https://github.com/a227799770055/CVmarathon_50days/blob/main/D2/Day002_change_color_space_HW.ipynb)
- [x] Day03 顏色相關預處理 [作業連結](https://github.com/a227799770055/CVmarathon_50days/blob/main/D3/Day003_color_spave_op_HW.ipynb)
- [x] Day04 圖片矩陣操作 [作業連結](https://github.com/a227799770055/CVmarathon_50days/blob/main/D4/Day004_geometric_transform_HW.ipynb)
- [x] Day05  [作業連結](https://github.com/a227799770055/CVmarathon_50days/blob/main/D5/Day005_draw_HW.ipynb)
- [x] Day06  [作業連結](https://github.com/a227799770055/CVmarathon_50days/blob/main/D6/Day006_affine_HW.ipynb)

- [x] Day18
```python
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import backend as K



def VGG16(include_top=True,input_tensor=None, input_shape=(224,224,1),
          pooling='max',classes=1000):
 
    img_input = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    '''可參考上面的搭法'''
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    '''可參考上面的搭法'''
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
  
    # Block 5
    '''可參考上面的搭法'''
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')

   
    return model


```










