import numpy as np
import matplotlib.pyplot as plt

train_dir = '/home/mcdonald/Dropbox/Cho/'
eval_history = np.load(train_dir+'eval_history924.npy')
loss_history = np.load(train_dir+'loss_history924.npy')
labels = ['teacher', 'large', 'medium', 'small']

print(eval_history.shape)
x = np.arange(eval_history.shape[1])*100.

plt.figure(1)
for i in range(4):
  plt.plot(x[:1000], eval_history[i][:1000], label=labels[i])

plt.ylim(0, 1)
plt.title('Accuracy on Validation Set')
plt.xlabel('Batches (128 images/batch)')
plt.ylabel('Accuracy')
plt.xticks(np.arange(10)*10000)
plt.legend(loc=0)
plt.show()

plt.figure(2)
for i in range(4):
  plt.plot(loss_history[i][:100000], label=labels[i])

plt.title('Value of Loss Function')
plt.xlabel('Batches (128 images/batch)')
plt.ylabel('Loss Function')
plt.legend(loc=0)
plt.show()


