import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
plt.show()

# 그레이하운드와 래브라도를 구분할 때, 키를이용하여 구분하는 것은 좋지 않은 방법이다.
# 때문에 더 좋은 분류법을 생각해 낼 필요가 있다.
