import numpy as np
for i in range(5):
    tf_loss = np.load('./probe_output/iter_{}/probe/cross_entropy_probe:0.npy'.format(i))
    print("iter_{}: {}".format(i,tf_loss))
