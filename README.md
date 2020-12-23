# CVmarathon_50days
### Computer Vision and Deep Learning Marathson




- [x] Day01 OpenCV 簡介與開啟圖片 [作業連結](https://github.com/a227799770055/CVmarathon_50days/blob/main/D1/Day01.ipynb)
- [x] Day02 Color Presentation [作業連結](https://github.com/a227799770055/CVmarathon_50days/blob/main/D2/Day002_change_color_space_HW.ipynb)
- [x] Day03 顏色相關預處理 [作業連結](https://github.com/a227799770055/CVmarathon_50days/blob/main/D3/Day003_color_spave_op_HW.ipynb)
- [x] Day04 圖片矩陣操作 [作業連結](https://github.com/a227799770055/CVmarathon_50days/blob/main/D4/Day004_geometric_transform_HW.ipynb)
- [x] ***Day05***

```python
img_hw = img.copy()
point1 = [60, 40]
point2 = [420, 510]

"""
對明亮度做直方圖均衡
"""
# 原始 BGR 圖片轉 HSV 圖片
img_hw = cv2.cvtColor(img_hw, cv2.COLOR_BGR2HSV)

# 對明亮度做直方圖均衡 -> 對 HSV 的 V 做直方圖均衡

# 將圖片轉回 BGR
img_hw = cv2.cvtColor(img_hw, cv2.COLOR_HSV2BGR)

"""
水平鏡像
"""
h, w = img_hw.shape[:2]

# 圖片鏡像
img_hw = img_hw[:,::-1,:]

# 透過四則運算計算鏡像後位置
# 確保點的位置一樣是左上跟右下，所以交換鏡像後的 x 座標 (y 座標做水平鏡像後位置不變)
point1[0] = h-point1[0]
point2[0] = h-point1[0]
print(point1, point2)
"""
縮放處理 (0.5 倍)
"""
fx = 0.5
fy = 0.5
resize_col = int(img_hw.shape[1]*fx)
resize_row = int(img_hw.shape[0]*fy)
print('resize_row={}, resize_col={}'.format(resize_row, resize_col))
print('img_hw shape ={}'.format(img_hw.shape))
# 建構 scale matrix
M_scale = np.array([[fx,0,0],
                   [0,fy,0]])
img_hw = cv2.warpAffine(img_hw,M_scale,(resize_row, resize_col))
# 把左上跟右下轉為矩陣型式
bbox = np.array((point1, point2), dtype=np.float32)
print('M_scale.shape={}, bbox.shape={}'.format(M_scale.shape, bbox.shape))

# 做矩陣乘法可以使用 `np.dot`, 為了做矩陣乘法, M_scale 需要做轉置之後才能相乘
homo_coor_result = np.dot(M_scale.T, bbox)
homo_coor_result = homo_coor_result.astype('uint8')
print(homo_coor_result)
scale_point1 = tuple(homo_coor_result[0])
scale_point2 = tuple(homo_coor_result[1])
print('origin point1={}, origin point2={}'.format(point1, point2))
print('resize point1={}, resize point2={}'.format(scale_point1, scale_point2))

# 畫圖
cv2.rectangle(img_hw, scale_point1, scale_point2, (0, 0, 255), 3)

while True:
    cv2.imshow('image', img_hw)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
    break
```
- [x] ***Day06***
```python
# 給定兩兩一對，共三對的點
# 這邊我們先用手動設定三對點，一般情況下會有點的資料或是透過介面手動標記三個點
rows, cols = img.shape[:2]
pt1 = np.array([[50,50], [300,100], [200,300]], dtype=np.float32)
pt2 = np.array([[80,80], [330,150], [300,300]], dtype=np.float32)

# 取得 affine 矩陣並做 affine 操作
M_affine = cv2.getAffineTransform(pt1,pt2,)
print(M_affine)
img_affine = cv2.warpAffine(img, M_affine,(img.shape[0],img.shape[1]))

# 在圖片上標記點
img_copy = img.copy()
for idx, pts in enumerate(pt1):
    pts = tuple(map(int, pts))
    cv2.circle(img_copy, pts, 3, (0, 255, 0), -1)
    cv2.putText(img_copy, str(idx), (pts[0]+5, pts[1]+5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

for idx, pts in enumerate(pt2):
    pts = tuple(map(int, pts))
    cv2.circle(img_affine, pts, 3, (0, 255, 0), -1)
    cv2.putText(img_affine, str(idx), (pts[0]+5, pts[1]+5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    
# 組合 + 顯示圖片
img_show_affine = np.hstack((img_copy, img_affine))
while True:
    cv2.imshow('affine transformation', img_show_affine)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
    break
```

















