
### 问题概述

- **错误思路**：一开始的想法是，通过整除一个步长值，再乘以该步长来获取灰度值，然而这会导致输出的灰度范围不完整。

  例如：  
  255 // 128 * 128 = 128  
  0 // 128 * 128 = 0  
  这样只能得到0灰度和128灰度。

- **正确思路**：为了全面利用灰度范围，我们应该将8bit灰度映射到最接近的对应bit灰度上。

- **额外注意**：例如，4bit可以表示4种灰度，但255应该除以(4-1)，而不是4。因为有n个灰度值（或n个点）只能划分出n-1个区间。

### 代码实现

```py
bits = 2
for i in imgs:
    for j in range(height):
        for k in range(width):
            parts = bits - 1
            length = 256 / parts
            halfLength = length / 2
            level = img[j, k] // length
            if img[j, k] > level * length + halfLength:
                i[j, k] = (level + 1) * length - 1
            else:
                i[j, k] = level * length
    bits = bits * 2
```
 