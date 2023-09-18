# 输入插值法的输入图像和插值法的4个输出图像，分别将输入图像减去4个输出图像，得到4个差值图像，将这七副图像显示在一起，观察差值图像的特点，并分别保存4幅差值图像。
# 分别是三种插值法图像，和一个ai超分辨率图像
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 输入路径如下
# digitalImageProcessing\ch02\insert\input.png
# digitalImageProcessing\ch02\insert\output_bicubic.png
# digitalImageProcessing\ch02\insert\output_bilinear.png
# digitalImageProcessing\ch02\insert\output_nn.png
# digitalImageProcessing\ch02\insert\ai_output.png

# 读取图像
img = cv2.imread('digitalImageProcessing\ch02\insert\input.png')
img_bicubic = cv2.imread('digitalImageProcessing\ch02\insert\output_bicubic.png')
img_bilinear = cv2.imread('digitalImageProcessing\ch02\insert\output_bilinear.png')
img_nn = cv2.imread('digitalImageProcessing\ch02\insert\output_nn.png')
img_ai = cv2.imread('digitalImageProcessing\ch02\insert\\ai_output.png')

# 将图像转换为RGB格式
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_bicubic = cv2.cvtColor(img_bicubic, cv2.COLOR_BGR2RGB)
img_bilinear = cv2.cvtColor(img_bilinear, cv2.COLOR_BGR2RGB)
img_nn = cv2.cvtColor(img_nn, cv2.COLOR_BGR2RGB)
img_ai = cv2.cvtColor(img_ai, cv2.COLOR_BGR2RGB)

# 计算差值图像
img_bicubic_diff = img - img_bicubic
img_bilinear_diff = img - img_bilinear
img_nn_diff = img - img_nn
img_ai_diff = img - img_ai

# 计算差值图像的均值
print(np.mean(img_bicubic_diff))
print(np.mean(img_bilinear_diff))
print(np.mean(img_nn_diff))
print(np.mean(img_ai_diff))

# 计算差值图像的方差
print(np.var(img_bicubic_diff))
print(np.var(img_bilinear_diff))
print(np.var(img_nn_diff))
print(np.var(img_ai_diff))

# 将均值和方差保存到文件
with open('digitalImageProcessing\ch02\subPicture\output.txt', 'w') as f:
    f.write('bicubic mean: ' + str(np.mean(img_bicubic_diff)) + 'bicubic var: ' + str(np.var(img_bicubic_diff)) + '\n')
    f.write('bilinear mean: ' + str(np.mean(img_bilinear_diff)) + 'bilinear var: ' + str(np.var(img_bilinear_diff)) + '\n')
    f.write('nn mean: ' + str(np.mean(img_nn_diff)) + 'nn var: ' + str(np.var(img_nn_diff)) + '\n')
    f.write('ai mean: ' + str(np.mean(img_ai_diff)) + 'ai var: ' + str(np.var(img_ai_diff)) + '\n')

# 显示图像
plt.figure(figsize=(15, 5))

plt.subplot(2, 4, 1)
plt.imshow(img)
plt.title('Input')

plt.subplot(2, 4, 2)
plt.imshow(img_bicubic)
plt.title('Bicubic')

plt.subplot(2, 4, 3)
plt.imshow(img_bilinear)
plt.title('Bilinear')

plt.subplot(2, 4, 4)
plt.imshow(img_nn)
plt.title('Nearest Neighbor')

plt.subplot(2, 4, 5)
plt.imshow(img_bicubic_diff)
plt.title('Input - Bicubic')

plt.subplot(2, 4, 6)
plt.imshow(img_bilinear_diff)
plt.title('Input - Bilinear')

plt.subplot(2, 4, 7)
plt.imshow(img_nn_diff)
plt.title('Input - Nearest Neighbor')

plt.subplot(2, 4, 8)
plt.imshow(img_ai_diff)
plt.title('Input - AI')

plt.show()

# 分别存储4幅图像
plt.imsave('digitalImageProcessing\ch02\subPicture\output_bicubic_diff.png', img_bicubic_diff)
plt.imsave('digitalImageProcessing\ch02\subPicture\output_bilinear_diff.png', img_bilinear_diff)
plt.imsave('digitalImageProcessing\ch02\subPicture\output_nn_diff.png', img_nn_diff)
plt.imsave('digitalImageProcessing\ch02\subPicture\\ai_output_diff.png', img_ai_diff)
