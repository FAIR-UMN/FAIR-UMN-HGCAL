#!/home/rusack/joshib/.conda/envs/fair_gpu/bin/python
import os,sys
from tqdm import tqdm
import pickle

import torch
from models.DNN import NN
from dataset_utils import H5DatasetDNN, split_dataset
from torch.utils.data.dataloader import default_collate
from utils.torch_utils import MARELoss, train

EPOCHS = 100
TRAIN_BATCH_SIZE = 10000
LAYERS = [51, 40, 30, 1] # shape of DNN
LRATE = 1e-2 # learning rate
TRAINING_FOLDER="../training/test_epochs_30_lr_0p01_bs_1e5"

if not os.path.exists(TRAINING_FOLDER):
        os.system('mkdir -p {}'.format(TRAINING_FOLDER))

file_path = '/home/rusack/shared/hdf5/hgcal_pion/hgcal_pions_combinedHgc_Ahc_1.h5'
dataset = H5DatasetDNN(file_path)
#data_indices = range(len())
train_test_datasets = split_dataset(dataset)

# check if gpus are available
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0'

X = train_test_dataset['train']
Y = train_test_datasets['test']

dataloaders = { 'train': torch.utils.data.DataLoader(X, TRAIN_BATCH_SIZE, shuffle=True,
                         collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))),
                'test': torch.utils.data.DataLoader(Y, len(Y), shuffle=True,
                        collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))}

# Initialize the network
nn = NN(LAYERS).to(device)
optimizer = torch.optim.Adam(nn.parameters(), lr = LRATE)
loss_func = MARELoss

epochs = []
lr_array = []
train_loss_array = []
valid_loss_array = []

pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    for xtrain, ytrain in dataloaders['train']:
        xtrain = xtrain
        train_loss, output_train = train(nn, xtrain, ytrain, optimizer, loss_func)
        
        test_loss = None
        output_test = None
        with torch.no_grad():
            for xtest, ytest in dataloaders['test']:
                xtest = torch.reshape(nn(xtest), (-1,))
                test_loss = MARELoss(xtest, ytest)
        
        epochs.append(epoch)
        train_loss_array.append(train_loss.item())
        valid_loss_array.append(test_loss.item())
        lr_array.append(optimizer.param_groups[0]['lr'])

        pbar.set_postfix({'training loss': train_loss.item(), 'validation loss': test_loss.item()})
        torch.save(nn.state_dict(), f'{TRAINING_FOLDER}/epoch{epoch}')

training_summary = {
    'epochs': epochs,
    'train_loss': train_loss_array,
    'valid_loss': valid_loss_array,
    'learning_rate': lr_array
}

with open(f'{TRAINING_FOLDER}/summary.pkl','wb') as f_:
    pickle.dump(training_summary, f_)


file_path = '/home/rusack/shared/hdf5/hgcal_electron/flat_0001/hgcal_electron_data_0001.h5'
dataset = H5DatasetDNN(file_path)
dataloaders['full'] torch.utils.data.DataLoader(dataset, TRAIN_BATCH_SIZE, shuffle=False)

nn.load_state_dict(torch.load('../training/test_epochs_30_lr_0p01_bs_1e5/epoch29', map_location=torch.device('cpu')))
nn_output = None
combined_output = []
with torch.no_grad():
    for xtest, ytest in dataloaders['test']:
        nn_output = nn(xtest).reshape(-1,)

