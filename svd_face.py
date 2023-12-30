import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io

face_contents = scipy.io.loadmat(os.path.join('.', 'allFaces.mat'))

faces = face_contents['faces']
m = int(face_contents['m'])
n = int(face_contents['n'])

nfaces = np.ndarray.flatten(face_contents['nfaces'])

training_faces = faces[:,:np.sum(nfaces[:36])]
avg_face = np.mean(training_faces, axis=1)

X = training_faces - np.tile(avg_face, (training_faces.shape[1], 1)).T
U, S, VT = np.linalg.svd(X, full_matrices=0)

fig1, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig1.suptitle('First 3 Eigenfaces')

ax1.imshow(np.reshape(U[:,0],(m, n)).T).set_cmap('gray')
ax1.axis('off')

ax2.imshow(np.reshape(U[:,1],(m, n)).T).set_cmap('gray')
ax2.axis('off')

ax3.imshow(np.reshape(U[:,2],(m, n)).T).set_cmap('gray')
ax3.axis('off')

plt.show()

test_face = faces[:,np.sum(nfaces[:36])]
test_face_centered = test_face - avg_face

recon_test_face = avg_face + U[:,:m*n]@U[:,:m*n].T@test_face_centered

fig2, (ax4, ax5) = plt.subplots(1, 2)
fig1.suptitle('eface Reconstruction (800)')

ax4.imshow(np.reshape(test_face,(m, n)).T).set_cmap('gray')
ax4.axis('off')

ax5.imshow(np.reshape(recon_test_face,(m, n)).T).set_cmap('gray')
ax5.axis('off')

plt.show()