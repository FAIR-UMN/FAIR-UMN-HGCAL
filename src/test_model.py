#!/home/rusack/joshib/.conda/envs/fair_gpu/bin/python
import os,sys
from tqdm import tqdm
import pickle

import torch
from models.DNN import NN
from dataset_utils import H5DatasetDNN, PklDatasetDNN, split_dataset
from torch.utils.data.dataloader import default_collate
from utils.torch_utils import MARELoss, train

EPOCHS = 30
TRAIN_BATCH_SIZE = 10000
LAYERS = [28, 40, 35, 1] # shape of DNN
LRATE = 1e-2 # learning rate
TRAINING_FOLDER="../training/test_epochs_30_lr_0p01_bs_1e5"

path_to_pickles = '/home/rusack/shared/pickles/hgcal_electron/data_h3_selection/100'

if not os.path.exists(TRAINING_FOLDER):
        os.system('mkdir -p {}'.format(TRAINING_FOLDER))

# check if gpus are available
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0'

# Initialize the network
nn = NN(LAYERS).to(device)
optimizer = torch.optim.Adam(nn.parameters(), lr = LRATE)
loss_func = MARELoss

epochs = []
lr_array = []
train_loss_array = []
valid_loss_array = []

nn.load_state_dict(torch.load('../training/test_epochs_30_lr_0p01_bs_1e5/epoch29', map_location=torch.device('cpu')))
nn_output = None
dataset = PklDatasetDNN(path_to_pickles)
dl = torch.utils.data.DataLoader(dataset, len(dataset), shuffle=False)

nn_output = None

with torch.no_grad():
   for xtest, ytest in dl:
      nn_output = nn(xtest.float())

path_to_dir = '/'.join(path_to_pickles.split('/')[:-1])

with open(path_to_dir+'/dnn_output.pickle','wb') as f_:
    pickle.dump(np.asarray(nn_output), f_)
