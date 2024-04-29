import h5py
import numpy as np

# Get OpenFace_2 data
f = h5py.File("cmumosei/mosei/mosei.hdf5")

all_keys = list(f["OpenFace_2"].keys())
all_openface = [np.array(f["OpenFace_2"][k]["features"]) for k in all_keys]
all_labels = [np.array(f["All Labels"][k]["features"]) for k in all_keys]

all_labels = [np.repeat(label, repeats=face.shape[0], axis=0) for label, face in zip(all_labels, all_openface)]

all_openface = np.concatenate(all_openface, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

print(all_openface.shape, all_labels.shape)

np.savez_compressed("openface_and_labels_by_frame.npz", x=all_openface, y=all_labels)
