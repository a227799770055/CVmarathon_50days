# CVmarathon_50days
### Computer Vision and Deep Learning Marathson




- [x] Day01 OpenCV 簡介與開啟圖片 [作業連結](https://github.com/a227799770055/CVmarathon_50days/blob/main/D1/Day01.ipynb)
- [x] Day02 Color Presentation [作業連結](https://github.com/a227799770055/CVmarathon_50days/blob/main/D2/Day002_change_color_space_HW.ipynb)
- [x] Day03 顏色相關預處理 [作業連結](https://github.com/a227799770055/CVmarathon_50days/blob/main/D3/Day003_color_spave_op_HW.ipynb)
- [x] Day04 圖片矩陣操作 [作業連結](https://github.com/a227799770055/CVmarathon_50days/blob/main/D4/Day004_geometric_transform_HW.ipynb)
- [x] Day05  [作業連結](https://github.com/a227799770055/CVmarathon_50days/blob/main/D5/Day005_draw_HW.ipynb)
- [x] Day06  [作業連結](https://github.com/a227799770055/CVmarathon_50days/blob/main/D6/Day006_affine_HW.ipynb)
- [x] Day11
```python
##輸入照片尺寸==28*28*1
##都用一層，288個神經元

##建造一個一層的CNN層
classifier=Sequential()

##Kernel size 3*3，用32張，輸入大小28*28*1
classifier.add(Convolution2D(32,(3,3),input_shape=(28,28,1)))
'''32張Kernel，大小為3*3，輸入尺寸為28*28*1'''
##看看model結構
print(classifier.summary())
'''理解輸出Total params為何==320'''

##建造一個一層的FC層
classifier=Sequential()
##輸入為28*28*1攤平==784
inputs = Input(shape=(784,))
'''輸入尺寸為28*28*1'''
##CNN中用了(3*3*1)*32個神經元，我們這邊也用相同神經元數量
x=Dense(288)(inputs)
'''使用288個神經元'''
model = Model(inputs=inputs, outputs=x)
##看看model結構
print(model.summary())
'''理解輸出Total params為何==226080'''

##大家可以自己變換設定看看參數變化
```
- [x] Day12
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


##kernel size=(6,6)
##kernel數量：32

## Same padding、strides=(1,1)
classifier=Sequential()
inputs = Input(shape=(13,13,1))
x=Convolution2D(32,(6,6),padding='same',strides=(1,1))(inputs)
model = Model(inputs=inputs, outputs=x)
model.summary()
## Same padding、strides=(2,2)
classifier=Sequential()
inputs = Input(shape=(13,13,1))
x=Convolution2D(32,(6,6),padding='same',strides=(2,2))(inputs)
model = Model(inputs=inputs, outputs=x)
model.summary()
## Valid padding、strides=(1,1)
classifier=Sequential()
inputs = Input(shape=(13,13,1))
x=Convolution2D(32,(6,6),padding='valid',strides=(1,1))(inputs)
model = Model(inputs=inputs, outputs=x)
model.summary()
## Valid padding、strides=(2,2)
classifier=Sequential()
inputs = Input(shape=(13,13,1))
x=Convolution2D(32,(6,6),padding='valid',strides=(2,2))(inputs)
model = Model(inputs=inputs, outputs=x)
model.summary()
```
- [x] Day13
```python
input_shape = (32, 32, 3)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), padding='same',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))  ##pooling_size=2,2 strides=2,2 輸出feature map 大小為多少？

model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))##pooling_size=2,2 strides=2,2 輸出feature map 大小為多少？

model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(1,1), strides=(1,1)))##pooling_size=1,1 strides=1,1 輸出feature map 大小為多少？

model.add(Conv2D(10, kernel_size=(3, 3), padding='same'))
model.add(Flatten()) ##Flatten完尺寸如何變化？
#model.add(GlobalAveragePooling2D()) #關掉Flatten，使用GlobalAveragePooling2D，完尺寸如何變化？

model.add(Dense(28)) ##全連接層使用28個units
```

- [x] Day14
```python
input_shape = (32, 32, 3)

model = Sequential()

##  Conv2D-BN-Activation('sigmoid') 

#BatchNormalization主要參數：
#momentum: Momentum for the moving mean and the moving variance.
#epsilon: Small float added to variance to avoid dividing by zero.

model.add(Conv2D(10,(3,3),input_shape=(32,32,3)))
model.add(BatchNormalization(momentum=0.99, epsilon=0.001)) 
model.add(Activation('sigmoid'))


##、 Conv2D-BN-Activation('relu')
model.add(Conv2D(10,(3,3)))
model.add(BatchNormalization(momentum=0.99, epsilon=0.001)) 
model.add(Activation('relu'))


model.summary()
```
- [x] Day15

```python
classifier=Sequential()

#卷積組合
classifier.add(Convolution2D(32,(3,3), input_shape=(32,32,3)))#32,3,3,input_shape=(32,32,3),activation='relu''
classifier.add(BatchNormalization(momentum=0.99,epsilon=0.001))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),padding='valid'))

classifier.add(Convolution2D(32,(3,3)))
classifier.add(BatchNormalization(momentum=0.99,epsilon=0.001))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),padding='valid'))

#flatten
classifier.add(Flatten())

#FC
classifier.add(Dense(128,activation='relu')) #output_dim=100,activation=relu
classifier.add(Dense(64,activation='relu'))
classifier.add(Dense(32,activation='relu'))
#輸出
classifier.add(Dense(10,activation='sigmoid'))

#超過兩個就要選categorical_crossentrophy
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(x_train,y_train,batch_size=100,epochs=10)

classifier.predict(x_test)
classifier.evaluate(x_test,y_test)
```












