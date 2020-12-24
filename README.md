# CVmarathon_50days
### Computer Vision and Deep Learning Marathson




- [x] Day01 OpenCV 簡介與開啟圖片 [作業連結](https://github.com/a227799770055/CVmarathon_50days/blob/main/D1/Day01.ipynb)
- [x] Day02 Color Presentation [作業連結](https://github.com/a227799770055/CVmarathon_50days/blob/main/D2/Day002_change_color_space_HW.ipynb)
- [x] Day03 顏色相關預處理 [作業連結](https://github.com/a227799770055/CVmarathon_50days/blob/main/D3/Day003_color_spave_op_HW.ipynb)
- [x] Day04 圖片矩陣操作 [作業連結](https://github.com/a227799770055/CVmarathon_50days/blob/main/D4/Day004_geometric_transform_HW.ipynb)

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















=======
- [x] Day05  [作業連結](https://github.com/a227799770055/CVmarathon_50days/blob/main/D5/Day005_draw_HW.ipynb)
- [x] Day06  [作業連結](https://github.com/a227799770055/CVmarathon_50days/blob/main/D6/Day006_affine_HW.ipynb)
>>>>>>> 02fc0e97be5dc0191bbfe722fdf24fd38bc3cefb

