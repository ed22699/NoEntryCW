import matplotlib.pyplot as plt
import numpy as np
tpVals = [1, 1, 1]
fpVals = [1, 0.0164799, 0.000523583]
steps = [0, 1, 2]
plt.plot(steps, tpVals, label="TPR", color="green")
plt.plot(steps, fpVals, label="FPR", color="red")
plt.legend()
plt.title("The TPR and FPR on the different stages of boosting")
plt.xlabel("Training Stage")
plt.ylabel("Rate")
plt.xticks(np.arange(min(steps), max(steps)+1, 1))
plt.show()

