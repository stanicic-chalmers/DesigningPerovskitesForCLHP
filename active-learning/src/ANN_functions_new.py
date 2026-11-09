from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np
import random
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 18:19:04 2022

@author: p.tung; z.rao
"""

class FeatureDataset(Dataset): #from numpy to tensor (pytroch-readable)
    '''
    Args: x is a 2D numpy array [x_size, x_features]
    '''
    def __init__(self, x):
        self.x = x
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.x[idx])#, torch.FloatTensor(self.y[idx])
    
class AttributeDataset(Dataset): # this is for classifier 
    '''
    Args: x is a 2D numpy array [x_size, x_features]
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return torch.Tensor(self.x[idx]), torch.Tensor(self.y[idx])
    



"""
Cases for different decoders
 only WAE no regression decoder
 decoder with one property
 decoder with multiple propertes
 different sizes of sublattices 

Example of params dictionary, this one includes both parameters for training and 
how the actual decoder class works.

params = {
    # Training params first
    'num_epoch' : 500,
    'batch_size' : 32,
    'lr' : 1e-4,
    'weight_decay' : 0.0,
    'sigma' : 1.0,
    'MMD_lambda' : 1e-3,

    # Network architecture parameters below this are the mandatory parameters
    
    'latent_dim' : 8,
    'use_regression' : True, # This option can disable regression decoder
    'lattice_A_size' : 7,
    'lattice_B_size' : 7,
    'n_properties' : 3, # Number of propertes to predict with regression decoder
    # If 'use_regression' is False, n_propertes = 0
    # scale 1 of layer means layers have size [80,64,48]*scale
    'decoder_scale' : 4,
    'regression_scale' :4,
} # for WAE training
"""

class WAE_modular(nn.Module):
    def __init__(self, params):
        super(WAE_modular, self).__init__()


        # --- These will be added to params ---------
        self.use_regression = params['use_regression']
        self.A_size = params['lattice_A_size']
        self.B_size = params['lattice_B_size']
        self.n_properties = params['n_properties']
        self.end = self.A_size + self.B_size
        # Enable full dropout for MC Dropout uncertainty estimation
        self.use_full_dropout = params.get('use_full_dropout', False)
        # ------------------------------------

        input_size = self.A_size + self.B_size+self.n_properties
        latent_dim = params['latent_dim']

        # Layer-specific dropout rates for better uncertainty estimation
        self.encoder_dropout = params.get('encoder_dropout', 0.1)
        self.decoder_dropout = params.get('decoder_dropout', 0.1)
        self.regression_dropout = params.get('regression_dropout', 0.1)
        self.activation = nn.SiLU()
        
        decoder_scale = max(1, int(params.get('decoder_scale', 1)))
        self.layers = [max(1, int(x * decoder_scale)) for x in [80, 64, 48]]

        if self.use_regression: # Create layers for regression decoder and scale
            regression_scale = max(1, int(params.get('regression_scale', 1)))
            self.layers2 = [max(1, int(x * regression_scale)) for x in [80, 64, 48]]
        else:
            self.layers2 = None

        # Print layer sizes and dropout rates for quick verification when the model is instantiated
        print(f"WAE_modular init: decoder_scale={decoder_scale}, decoder layers={self.layers}")
        if self.use_regression:
            print(f"WAE_modular init: regression_scale={regression_scale}, regression layers={self.layers2}")
        print(f"WAE_modular init: use_full_dropout={self.use_full_dropout} (for MC Dropout)")
        print(f"WAE_modular init: dropout rates - encoder={self.encoder_dropout:.2f}, decoder={self.decoder_dropout:.2f}, regression={self.regression_dropout:.2f}")
        
        
        # encoder - Pre-LN pattern: Linear → LayerNorm → Activation
        encoder_layers = [
            nn.Linear(input_size, self.layers[0]),
            nn.LayerNorm(self.layers[0]),
            self.activation,
        ]
        if self.use_full_dropout or True:  # Always add dropout after first layer
            encoder_layers.append(nn.Dropout(self.encoder_dropout))
        encoder_layers.append(nn.Linear(self.layers[0], self.layers[1]))
        encoder_layers.append(nn.LayerNorm(self.layers[1]))
        encoder_layers.append(self.activation)
        if self.use_full_dropout:
            encoder_layers.append(nn.Dropout(self.encoder_dropout))
        encoder_layers.extend([
            nn.Linear(self.layers[1], self.layers[2]),
            nn.LayerNorm(self.layers[2]),
            self.activation,
        ])
        if self.use_full_dropout:
            encoder_layers.append(nn.Dropout(self.encoder_dropout))
        encoder_layers.append(nn.Linear(self.layers[2], latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # decoder - Pre-LN pattern: Linear → LayerNorm → Activation
        composition_decoder_layers = [
            nn.Linear(latent_dim, self.layers[2]),
            nn.LayerNorm(self.layers[2]),
            self.activation,
        ]
        if self.use_full_dropout or True:  # Always add dropout after first layer
            composition_decoder_layers.append(nn.Dropout(self.decoder_dropout))
        composition_decoder_layers.append(nn.Linear(self.layers[2], self.layers[1]))
        composition_decoder_layers.append(nn.LayerNorm(self.layers[1]))
        composition_decoder_layers.append(self.activation)
        if self.use_full_dropout:
            composition_decoder_layers.append(nn.Dropout(self.decoder_dropout))
        composition_decoder_layers.extend([
            nn.Linear(self.layers[1], self.layers[0]),
            nn.LayerNorm(self.layers[0]),
            self.activation,
        ])
        if self.use_full_dropout:
            composition_decoder_layers.append(nn.Dropout(self.decoder_dropout))
        composition_decoder_layers.append(nn.Linear(self.layers[0], input_size-self.n_properties))
        # (input size = 17, n_props = 3, output is 14)
        self.composition_decoder = nn.Sequential(*composition_decoder_layers)
        
        # regression decoder - Pre-LN pattern: Linear → LayerNorm → Activation
        if self.use_regression:
            regression_decoder_layers = [
                nn.Linear(latent_dim, self.layers2[2]),
                nn.LayerNorm(self.layers2[2]),
                self.activation,
            ]
            if self.use_full_dropout or True:  # Always add dropout after first layer
                regression_decoder_layers.append(nn.Dropout(self.regression_dropout))
            regression_decoder_layers.append(nn.Linear(self.layers2[2], self.layers2[1]))
            regression_decoder_layers.append(nn.LayerNorm(self.layers2[1]))
            regression_decoder_layers.append(self.activation)
            if self.use_full_dropout:
                regression_decoder_layers.append(nn.Dropout(self.regression_dropout))
            regression_decoder_layers.extend([
                nn.Linear(self.layers2[1], self.layers2[0]),
                nn.LayerNorm(self.layers2[0]),
                self.activation,
            ])
            if self.use_full_dropout:
                regression_decoder_layers.append(nn.Dropout(self.regression_dropout))
            regression_decoder_layers.append(nn.Linear(self.layers2[0], self.n_properties))
            self.regression_decoder = nn.Sequential(*regression_decoder_layers)
            self.apply(weights_init)

    def forward(self, x):
        z = self._encode(x)

        if self.use_regression:
            decoded = self._decode_regression(z)
        else:
            decoded = self._decode(z)
        
        return decoded, z
    
    def _encode(self, x):
        return self.encoder(x)
    
    def _decode(self,z):
        compositions = self.composition_decoder(z)
        # Add separete postprocessing for the two sublattices
        sublattice_A = F.softmax(compositions[:, 0:self.A_size], dim=1)
        sublattice_B = F.softmax(compositions[:, self.A_size:], dim=1)
        return torch.cat((sublattice_A, sublattice_B), dim=1)

    def _decode_regression(self, z):
        compositions = self.composition_decoder(z)
        # Add separete postprocessing for hpp and the two sublattices
        sublattice_A = F.softmax(compositions[:, 0:self.A_size], dim=1)
        sublattice_B = F.softmax(compositions[:, self.A_size:], dim=1)
        # Indeces are correct above note that they are the ones from the compostion decoder

        hpp = self.regression_decoder(z)  # hpp is first
        x_recon = torch.cat((hpp, sublattice_A, sublattice_B), dim=1)

        return x_recon
    

# a MLP for regression of hpp from 
class regression_MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        scale = 1
        self.activation = nn.ReLU()
        self.layers = [80,64,48]
        self.layers = [int(x * scale) for x in self.layers]
        self.dropout_rate = 0.10
        # encoder

        self.mlp = nn.Sequential(
                            nn.Linear(self.input_size, self.layers[0]),
                            nn.Dropout(self.dropout_rate),
                            nn.LayerNorm(self.layers[0]),
                            self.activation,
                            nn.Linear(self.layers[0], self.layers[1]),
                            nn.LayerNorm(self.layers[1]),
                            self.activation,
                            nn.Linear(self.layers[1], self.layers[2]),
                            nn.LayerNorm(self.layers[2]),
                            self.activation,
                            nn.Linear(self.layers[2], 1)
                            #nn.Softmax(dim=1) #(softmad along dimension 1)
                            )
        self.apply(weights_init)

    def forward(self, x):
        return self.mlp(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
def same_seeds(seed): #fix np & torch seed to the same.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # np.random.seed(seed)  # Numpy module.
    # random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_latents(model, dataset): #from dataset to altten
    model.to(device).eval() # training model or evaluation mode, eval means setting the model to its evaluation mode (gradient fixed)
    latents = []
    with torch.no_grad(): # fix the gradient, assure that the model parameters are fixed
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
        for i, data in enumerate(dataloader):
            x = data.to(device)
            recon_x, z = model(x)
            latents.append(z.detach().cpu().numpy())
    return np.concatenate(latents,axis=0)



def imq_kernel(X: torch.Tensor, Y: torch.Tensor, h_dim: int): # common kerntl to choose
    batch_size = X.size(0)

    # Precompute all distances once using more efficient cdist
    dists_x = torch.cdist(X, X, p=2).pow(2)  # distances within x cluster
    dists_y = torch.cdist(Y, Y, p=2).pow(2)  # distances within y cluster
    dists_c = torch.cdist(X, Y, p=2).pow(2)  # distances between x and y cluster

    # Create eye mask once
    eye_mask = 1 - torch.eye(batch_size, device=X.device)

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]: # Multiple kernel scales used, might be redundant for this application
        C = 2 * h_dim * 1.0 * scale

        # Compute kernel values for within-cluster distances
        res1 = C / (C + dists_x) + C / (C + dists_y)
        res1 = (eye_mask * res1).sum() / (batch_size - 1)

        # Compute kernel values for cross-cluster distances
        res2 = (C / (C + dists_c)).sum() * 2. / batch_size

        stats += res1 - res2

    return stats



# Like normal training nut returns all types of loss
def train_WAE(model, optimizer, dataloader, params, val_dataloader=None, logger=None):
    num_epoch = params['num_epoch']
    sigma = params['sigma'] # assuming the latent space follows Gaussian
    MMD_lambda = params['MMD_lambda'] #WAE distance (maximum mean discrepancy)
    latent_dim = params['latent_dim']
    max_grad_norm = params.get('max_grad_norm', 1.0)  # Gradient clipping threshold

    # Early stopping and checkpointing parameters
    early_stopping_patience = params.get('early_stopping_patience', None)  # None = no early stopping
    lr_patience = params.get('lr_patience', 100)  # Default patience for LR scheduler
    checkpoint_path = params.get('checkpoint_path', None)  # None = no checkpointing
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, 
        patience=lr_patience)

    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # folder_dir = os.path.join(script_dir, model_name) # a folder to save models
    # print(folder_dir)
    # if not os.path.isdir(folder_dir):
    #     os.mkdir(folder_dir)
    loss_=[]
    for epoch in range(num_epoch):
        #start_time = time.time()
        total_loss = 0.0
        total_recon = 0.0
        total_MMD = 0.0
        total_clas = 0.0

        for i, data in enumerate(dataloader): # Optimizer is ran after each batch
            x = data.to(device)
            model.train() # model goes to train mode
            recon_x, z_tilde = model(x) # latent space is Z_tilde
            z = sigma*torch.randn(z_tilde.size()).to(device) # z is sampled from a Gaussian that has the same dimension (but no relation to z_tilde).
            
            # Calculate reconstruction loss only for non-zero values
            #recon_loss = F.mse_loss(masked_recon_x, masked_x, reduction='mean')            #recon_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')
            #recon_loss = F.mse_loss(recon_x, x, reduction='mean') 
            recon_loss = F.l1_loss(recon_x, x, reduction='mean') # Seems best for our problem
            
            MMD_loss = imq_kernel(z_tilde, z, h_dim=latent_dim).to(device) #W-distance between z_tilde and z
            MMD_loss = MMD_loss / x.size(0) #averaging, because recon loss is mean.
            loss = recon_loss + MMD_loss * MMD_lambda #MM_lambda: learning-rate alike, hyperparamer
            #loss = MMD_loss*MMD_lambda
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_MMD += MMD_loss.item()

            total_clas += torch.abs(x[:, 0] - recon_x[:, 0]).mean().item()

            # Print the comparison of the first column of x and x_recon
            #print(f'Epoch: {epoch+1:03}, Iteration: {i+1:03}, Column 1 difference: {col_diff:.6f}')

    

        # Training averages
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_MMD = total_MMD / len(dataloader)
        avg_clas = total_clas/len(dataloader)

        # Validation Phase
        if val_dataloader is not None:
            model.eval()
            total_val_loss = 0.0
            total_val_recon = 0.0
            total_val_MMD = 0.0
            total_val_clas = 0.0

            with torch.no_grad():
                for i, x_val in enumerate(val_dataloader):
                    x_val = x_val.to(device)
                    recon_x_val, z_tilde_val = model(x_val)
                    z_val = sigma * torch.randn(z_tilde_val.size()).to(device)

                    # Compute validation losses (same as training)
                    recon_loss_val = F.l1_loss(recon_x_val, x_val, reduction='mean')
                    MMD_loss_val = imq_kernel(z_tilde_val, z_val, h_dim=latent_dim).to(device)
                    MMD_loss_val = MMD_loss_val / x_val.size(0)
                    loss_val = recon_loss_val + MMD_loss_val * MMD_lambda

                    # Accumulate validation metrics
                    total_val_loss += loss_val.item()
                    total_val_recon += recon_loss_val.item()
                    total_val_MMD += MMD_loss_val.item()
                    total_val_clas += torch.abs(x_val[:, 0] - recon_x_val[:, 0]).mean().item()

            # Compute validation averages
            avg_val_loss = total_val_loss / len(val_dataloader)
            avg_val_recon = total_val_recon / len(val_dataloader)
            avg_val_MMD = total_val_MMD / len(val_dataloader)
            avg_val_clas = total_val_clas / len(val_dataloader)

            # Store both training and validation metrics
            loss_.append([avg_loss, avg_recon, avg_MMD, avg_clas,
                         avg_val_loss, avg_val_recon, avg_val_MMD, avg_val_clas])

            # Check for improvement and save checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0

                # Save checkpoint if path is provided
                if checkpoint_path is not None:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'loss_history': loss_,
                    }, checkpoint_path)
                    if logger:
                        logger.info(f'Checkpoint saved at epoch {epoch+1} with val_loss: {avg_val_loss:.6f}')
            else:
                epochs_without_improvement += 1

            if epoch % 50 == 0 or epoch == num_epoch - 1:
                msg = '[{:03}/{:03}] Train Loss: {:.6f}, Recon: {:.6f}, MMD: {:.6f}, Val Loss: {:.6f}, Val Recon: {:.6f}, Val MMD: {:.6f}'.format(
                    epoch+1, num_epoch, avg_loss, avg_recon, avg_MMD, avg_val_loss, avg_val_recon, avg_val_MMD)
                msg2 = 'Last clas difference (train/val): {} {}'.format(avg_clas, avg_val_clas)
                msg += f' | LR: {optimizer.param_groups[0]["lr"]:.6e}'
                if logger:
                    logger.info(msg)
                    logger.info(msg2)
                else:
                    print(msg)
                    print(msg2)

            # Step scheduler with validation loss
            scheduler.step(avg_val_loss)

            # Early stopping check
            if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                msg = f'Early stopping triggered after {epoch+1} epochs (no improvement for {early_stopping_patience} epochs)'
                if logger:
                    logger.info(msg)
                else:
                    print(msg)
                break
        else:
            # Store only training metrics if no validation
            loss_.append([avg_loss, avg_recon, avg_MMD, avg_clas])

            # Check for improvement based on training loss (when no validation available)
            if avg_loss < best_val_loss:  # Using best_val_loss variable for training loss tracking
                best_val_loss = avg_loss
                epochs_without_improvement = 0

                # Save checkpoint if path is provided
                if checkpoint_path is not None:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_train_loss': best_val_loss,
                        'loss_history': loss_,
                    }, checkpoint_path)
                    if logger:
                        logger.info(f'Checkpoint saved at epoch {epoch+1} with train_loss: {avg_loss:.6f}')
            else:
                epochs_without_improvement += 1

            # Step scheduler with training loss
            scheduler.step(avg_loss)

            if epoch % 50 == 0 or epoch == num_epoch - 1:
                msg = '[{:03}/{:03}] loss: {:.6f} Recon_loss: {:.6f}, MMD_loss:{:.6f}'.format(
                    epoch+1, num_epoch, avg_loss, avg_recon, avg_MMD)
                msg2 = 'Last clas difference: {}'.format(avg_clas)
                
                msg += f' | LR: {optimizer.param_groups[0]["lr"]:.6e}'
                if logger:
                    logger.info(msg)
                    logger.info(msg2)
                else:
                    print(msg)
                    print(msg2)

            # Early stopping check (based on training loss if no validation)
            if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                msg = f'Early stopping triggered after {epoch+1} epochs (no improvement for {early_stopping_patience} epochs)'
                if logger:
                    logger.info(msg)
                else:
                    print(msg)
                break

    return loss_

# OBS this is untested for parametertests at the moment.
def train_WAE_testing_validation(model, optimizer, train_dataloader, val_dataloader, params):
    num_epoch = params['num_epoch']
    sigma = params['sigma']  # assuming the latent space follows Gaussian
    MMD_lambda = params['MMD_lambda']  # WAE distance (maximum mean discrepancy)
    latent_dim = params['latent_dim']

    loss_ = []
    val_loss = []
    for epoch in range(num_epoch):
        # start_time = time.time()
        total_loss = 0.0
        total_recon = 0.0
        total_MMD = 0.0
        total_clas = 0.0

        # Training Phase
        model.train()
        for i, data in enumerate(train_dataloader):
            x = data.to(device)
            #print(x)
            recon_x, z_tilde = model(x)  # latent space is Z_tilde

            z = sigma * torch.randn(z_tilde.size()).to(device)  # z is sampled from a Gaussian that has the same dimension (but no relation to z_tilde).

            recon_loss = F.l1_loss(recon_x, x, reduction='mean')  # Reconstruction loss
            #print(recon_x,x)
            #print(recon_loss)
            MMD_loss = imq_kernel(z_tilde, z, h_dim=latent_dim).to(device)  # W-distance between z_tilde and z
            MMD_loss = MMD_loss / x.size(0)  # averaging, because recon loss is mean.
            loss = recon_loss + MMD_loss * MMD_lambda  # Total loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_MMD += MMD_loss.item()
            total_clas += torch.abs(x[:, 0] - recon_x[:, 0]).mean().item()

        avg_loss = total_loss / len(train_dataloader)
        avg_recon = total_recon / len(train_dataloader)
        avg_MMD = total_MMD / len(train_dataloader)
        avg_clas = total_clas / len(train_dataloader)

       # Validation Phase
        model.eval()
        val_loss = 0.0
        for i, data in enumerate(val_dataloader):
            x = data.to(device)
            recon_x, z_tilde = model(x)
            z = sigma * torch.randn(z_tilde.size()).to(device)
            MMD_loss = imq_kernel(z_tilde, z, h_dim=latent_dim).to(device)
            MMD_loss = MMD_loss / x.size(0)
            val_loss += (recon_loss + MMD_loss * MMD_lambda).item()

        avg_val_loss = val_loss / len(val_dataloader)

        loss_.append([avg_loss, avg_recon, avg_MMD, avg_clas, avg_val_loss])

        if epoch % 50 == 0:
            print('[{:03}/{:03}] Train Loss: {:.6f}, Recon Loss: {:.6f}, MMD Loss: {:.6f}, Val Loss: {:.6f}'.format(
                epoch + 1, num_epoch, avg_loss, avg_recon, avg_MMD, avg_val_loss))

    return loss_
