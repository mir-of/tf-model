import numpy as np
loss = []
for i in range(10):
    tf_loss = np.load('./probe_output/iter_{}/probe/cross_entropy_probe:0.npy'.format(i))
    print("iter_{}: {}".format(i,tf_loss))
    loss.append(tf_loss)

np.save("1n1c", loss)
