import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读入图像
image = cv2.imread('digitalImageProcessing\ch02\stackNoiseReduction\input.png')
# 将BGR转为RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 生成100副图像
images = []
for i in range(100):
    # 生成高斯噪声
    noise = np.random.normal(0, 20, image.shape)
    # 图像加噪声
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    # 将噪声图像加入列表
    images.append(noisy_image)

# 将列表转换为numpy数组
images = np.array(images)

# 对图像数组进行堆栈处理
stacked_image = np.median(images, axis=0).astype(np.uint8)

# 显示原图像
plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(image)
plt.axis('off')

# 显示其中1副噪声图像
plt.subplot(1, 3, 2)
plt.title('Noisy Image')
plt.imshow(images[0])
plt.axis('off')

# 显示堆栈处理后的图像
plt.subplot(1, 3, 3)
plt.title('Stacked Image')
plt.imshow(stacked_image)
plt.axis('off')

# 保存图像
plt.tight_layout()
plt.savefig('digitalImageProcessing\ch02\stackNoiseReduction\stacked.png')
plt.show()
