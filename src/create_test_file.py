import h5py
import awkward as ak
import numpy as np

test_nevents = 1000

path_to_dir = '../data/'
file_path = '%s/flat_0001/hgcal_electron_data.h5' % path_to_dir
test_file_path = '%s/hgcal_electron_data_test.h5' % path_to_dir


h5arr = h5py.File(file_path, 'r')

column_list = ['rechit_x', 'rechit_y', 'rechit_z', 'rechit_energy', 'target', 'nhits']

arrays = {}
for cl in column_list:
    arrays[cl] = np.asarray(h5arr[cl])

test_arrays = {}

test_size = int(ak.sum(arrays['nhits'][:test_nevents]))

test_arrays['nhits'] = arrays['nhits'][:test_nevents]
test_arrays['target'] = arrays['target'][:test_nevents]

print(test_arrays['nhits'])

h5out = h5py.File(test_file_path, 'w')

h5out.create_dataset('nhits', data=test_arrays['nhits'], dtype=float)
h5out.create_dataset('target', data=test_arrays['target'], dtype=float)

for cl in column_list:
    if cl=='nhits' or cl=='target': continue
    test_arrays[cl] = arrays[cl][:test_size]
    h5out.create_dataset(cl, data=test_arrays[cl], dtype=float)

h5out.close()
h5arr.close()


print('Test file created at: %s' % test_file_path)
