import numpy as np
import cv2  # OpenCV 用於圖片讀取和處理
from scipy.fftpack import dct, idct  # 引入 scipy.fftpack 模組
from google.colab.patches import cv2_imshow  # 用於顯示圖片

# 讀取 lena_gray.bmp 圖片
image_path = "./Original_lena512.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 確保圖片是正方形並進行裁剪或縮放
n = 512  # 使用 512x512 大小
image = cv2.resize(image, (n, n))

# 顯示原始圖像
cv2_imshow(image)

# 設定每塊的大小
block_size = 8

# 取得圖片的尺寸
height, width = image.shape

# 創建一個空列表來儲存所有 8x8 塊的 DCT 結果
dct_blocks = []

# 拆解圖片為 8x8 的區塊，並計算每個區塊的 DCT
for i in range(0, height, block_size):
    for j in range(0, width, block_size):
        block = image[i:i+block_size, j:j+block_size]

        # 計算 8x8 區塊的 DCT
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

        # 保留 DCT 前 4x4 項，將後 4x4 項設為 0
        dct_block[4:, :] = 0  # 清除後 4 行
        dct_block[:, 4:] = 0  # 清除後 4 列

        dct_blocks.append(dct_block)

        # 輸出修改過的 DCT 區塊的矩陣
        #print(f"DCT Block ({i}, {j}):")
        #print(dct_block)

# 若需要將處理後的 DCT 塊進行 IDCT 還原
idct_blocks = []
for i, block in enumerate(dct_blocks):
    # 計算 IDCT 還原
    idct_block = idct(idct(block.T, norm='ortho').T, norm='ortho')
    idct_blocks.append(idct_block)

    # 輸出 IDCT 還原的結果矩陣
    #print(f"IDCT Block ({i // (width // block_size)}, {i % (width // block_size)}):")
    #print(idct_block)

# 重建圖片並顯示
idct_image = np.zeros_like(image)
for i in range(0, height, block_size):
    for j in range(0, width, block_size):
        # 還原每個區塊到對應位置
        idct_image[i:i+block_size, j:j+block_size] = idct_blocks.pop(0)

# 顯示並保存結果圖像
cv2_imshow(idct_image)
cv2.imwrite("lena_dct_idct_4x4.jpg", idct_image)


