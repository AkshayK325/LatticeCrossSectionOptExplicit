
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error
from tqdm import tqdm
import torch.nn.functional as F
# torch.set_default_dtype(torch.float32)

# from scipy.spatial import ConvexHull
# from matplotlib.patches import Polygon

def set_seed(manualSeed):
  torch.backends.cudnn.deterministic = True # type: ignore
  torch.backends.cudnn.benchmark = False # type: ignore
  torch.manual_seed(manualSeed)
  torch.cuda.manual_seed(manualSeed)
  torch.cuda.manual_seed_all(manualSeed)
  np.random.seed(manualSeed)
  random.seed(manualSeed)
class Encoder(nn.Module):
    def __init__(self, encoderSettings):
        super(Encoder, self).__init__()
        set_seed(1234)

        input_dim = encoderSettings['inputDim']
        hidden_dim = encoderSettings['hiddenDim']
        latent_dim = encoderSettings['latentDim']
        num_layers = encoderSettings['numLayers'] # New: to use numLayers from nnSettings

        # Dynamically build layers
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1): # num_layers is now total hidden layers, first one already added
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Latent space layers
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_sigma = nn.Linear(hidden_dim, latent_dim)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0
        self.isTraining = False
        
    
    def encode_base(self, x):
        h = x
        for layer in self.layers:
            h = F.leaky_relu(layer(h))
        mu = self.linear_mu(h)
        sigma = torch.exp(self.linear_sigma(h))
        return mu, sigma


    def forward(self, x):
        # Pass through hidden layers with ReLU
        # for i, layer in enumerate(self.layers):
        #     x = F.leaky_relu(layer(x))
        
        # # Get mean and log-variance
        # mu = self.linear_mu(x)
        # sigma = torch.exp(self.linear_sigma(x))
        mu, sigma = self.encode_base(x)
        
        if self.isTraining:
          noise = 0.1 * torch.randn_like(mu)        # small extra noise
          self.z = mu + sigma*self.N.sample(mu.shape).to(mu.device) + noise
        else:
          self.z = mu
          
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

        return self.z

class Decoder(nn.Module):
    def __init__(self, decoderSettings):
        super(Decoder, self).__init__()

        latent_dim = decoderSettings['latentDim']
        hidden_dim = decoderSettings['hiddenDim']
        output_dim = decoderSettings['outputDim']
        num_layers = decoderSettings['numLayers'] # New: to use numLayers from nnSettings

        # Dynamically build layers
        self.layers = nn.ModuleList()
        # Input from latent space
        self.layers.append(nn.Linear(latent_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # Ensure the input latent tensor has the same device and dtype as the
        # decoder parameters to avoid dtype/device mismatch errors when calling
        # linear layers (mat1 and mat2 must share dtype/device).

        for i, layer in enumerate(self.layers):
            z = F.leaky_relu(layer(z))

        z = torch.sigmoid(self.output_layer(z)) # decoder op in range [0,1]
        # z = torch.tanh(self.output_layer(z)) # decoder op in range [-1,1]
        return z
        

# # Neural network PropertyPredictor
class PropertyPredictor(nn.Module):
    def __init__(self, nn_architecture):  # Add device parameter
        super().__init__()
        manualSeed = 96
        set_seed(manualSeed)
        
        self.layers = nn.ModuleList()
        self.bnLayer = nn.ModuleList()
        self.dropout = nn.ModuleList() # List of dropouts

        self.nn_architecture = nn_architecture
        input_dim = nn_architecture['inputDim']
        hidden_dim = nn_architecture['hiddenDim']
        num_layers = nn_architecture['numLayers']
        output_dim = nn_architecture['outputDim']

        self.dropout_p = nn_architecture['dropout'] # Dropout probability
       
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.bnLayer.append(nn.BatchNorm1d(hidden_dim))
        self.dropout.append(nn.Dropout(self.dropout_p))

        # Hidden layers
        for i in range(num_layers - 1):
            l = nn.Linear(hidden_dim, hidden_dim)
            nn.init.xavier_normal_(l.weight)
            nn.init.zeros_(l.bias)
            self.layers.append(l)
            self.bnLayer.append(nn.BatchNorm1d(hidden_dim))
            self.dropout.append(nn.Dropout(self.dropout_p))

        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        # Calculate total weights and biases
        self.total_weights, self.total_biases = self.calculate_total_params()
        self.activation = nn.LeakyReLU()

    
    def forward(self, z):# z input and prop output
        for i in range(len(self.layers) - 1):
            z = self.activation(self.bnLayer[i](self.layers[i](z)))
            z = self.dropout[i](z) # Apply dropout after activation and batchnorm

        prop = self.layers[-1](z)
        return prop
        
    def calculate_total_params(self):
        """Calculates total weights and biases in the network."""
        total_weights = 0
        total_biases = 0
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                total_weights += layer.weight.numel()
                total_biases += layer.bias.numel()
        return total_weights, total_biases
        
    
        
class VariationalAutoencoder(nn.Module):
    def __init__(self, architecture_config):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(architecture_config['encoder'])
        self.decoder = Decoder(architecture_config['decoder'])
    
        if architecture_config['predictor'] is not None:
            self.predictor = PropertyPredictor(architecture_config['predictor'])
        else:
            self.predictor = None
  
    def forward(self, x):
      
        if self.predictor is not None:
            # VAE with property predictor
            z = self.encoder(x[:, :-self.predictor.nn_architecture['outputDim']])
            x_recon = self.decoder(z)
            prop = self.predictor(z)
            x_recon = torch.cat((x_recon, prop), dim=1)
        else:
            z = self.encoder(x)
            x_recon = self.decoder(z)

        return x_recon

    def encode(self, x):
        
        if self.predictor is not None:
            # VAE with property predictor
            z = self.encoder(x[:, :-self.predictor.nn_architecture['outputDim']])
            
        else:
            z = self.encoder(x)
    
        return z
		
    def predict(self, z):

        if self.predictor is not None:
            # VAE with property predictor
            x_recon = self.decoder(z)
            prop = self.predictor(z)
            x_recon = torch.cat((x_recon, prop), dim=1)
        else:
            x_recon = self.decoder(z)
        
        return x_recon
        
    


class VariationalAutoencoderModel:
    """
    A generalized class for training and interacting with a Variational Autoencoder.
    It handles data preprocessing, model training, and latent space visualization.
    """
    def __init__(self, input_data, scaling_info, identifiers, architecture_config,useCPU=False):
        """
        Initializes the VariationalAutoencoderModel.
    
        Args:
          input_data (torch.Tensor): The preprocessed (normalized, log-transformed)
                                     training data for the VAE.
          scaling_info (dict): Dictionary containing scaling parameters (min/max)
                               and original indices for each feature.
          identifiers (dict): Dictionary containing identifying information for each data point,
                              e.g., 'Name', 'Shape', 'classID'.
          architecture_config (dict): Dictionary containing VAE architecture settings
                                 for encoder and decoder (inputDim, hiddenDim, latentDim, outputDim).
        """
       
        if(torch.cuda.is_available() and (useCPU == False) ):
            self.device = torch.device("cuda:0")
            print("Running on GPU")
        else:
            self.device = torch.device("cpu")
            torch.set_num_threads(18)  
            print("Running on CPU\n")
            print("Number of CPU threads PyTorch is using:", torch.get_num_threads())
        
        self.input_data = input_data.to(self.device)
        self.scaling_info = scaling_info
        self.identifiers = identifiers
        self.architecture_config = architecture_config
        try:
            self.target_dtype = self.input_data.dtype
        except Exception:
            self.target_dtype = None
        self.vaeNet = VariationalAutoencoder(architecture_config)
    
        self.vaeNet.to(self.device)
  
    def load_model_from_file(self, model_state):
        """
        Loads the state dictionary of the VAE network from a specified file.
    
        Args:
          file_name (str): The path to the saved model state dictionary.
        """
        # Load the state dict into the network
        self.vaeNet.load_state_dict(model_state)

        # Ensure model is on the intended device and has the same dtype as
        # the input data tensor. This avoids dtype/device mismatches when
        # later passing tensors (e.g. latent z) that are created from
        # `self.input_data` or elsewhere in the pipeline.
        

        if self.target_dtype is not None:
            # Move model to device and cast to input dtype
            self.vaeNet.to(device=self.device, dtype=self.target_dtype)
            print(f"Model loaded and cast to device={self.device}, dtype={self.target_dtype}")
        else:
            # Fallback: just move to device
            self.vaeNet.to(self.device)
            print(f"Model loaded and moved to device={self.device}")

        self.vaeNet.eval() # Set the model to evaluation mode
  
    def train_model(self, num_epochs, kl_factor, saved_net_path, learning_rate):
        """
        Trains the Variational Autoencoder model.
    
        Args:
          num_epochs (int): Number of training epochs.
          kl_factor (float): Scaling factor for the KL divergence loss.
          saved_net_path (str): Path to save the trained model's state dictionary.
          learning_rate (float): Learning rate for the Adam optimizer.
    
        Returns:
          dict: A dictionary containing the history of reconstruction loss, KL loss, and total loss.
        """
        optimizer = torch.optim.Adam(self.vaeNet.parameters(), learning_rate, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(num_epochs), eta_min=1e-8)
        convergence_history = {'reconLoss': [], 'klLoss': [], 'loss': []}
    
        self.vaeNet.encoder.isTraining = True
    
        print("Starting VAE model training...")
        
        iterBar =True
        loop = tqdm(range(num_epochs), desc="Training Progress", ncols=75) if iterBar else range(num_epochs)
    
        mse = nn.MSELoss()
        # Wrap the training loop with tqdm
        min_kl = kl_factor
        for epoch in loop:
            optimizer.zero_grad()
            
            # kl_factor = min(min_kl, epoch / 20000)   # ramp for 20000 epochs
    
            pred_data = self.vaeNet(self.input_data)
            
            # KL divergence loss
            kl_loss = kl_factor * self.vaeNet.encoder.kl
            
            recon_loss = mse(pred_data, self.input_data)

            total_loss = recon_loss + kl_loss
    
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.vaeNet.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
    
            convergence_history['reconLoss'].append(recon_loss.item())
            convergence_history['klLoss'].append(self.vaeNet.encoder.kl.item()) # type: ignore
            convergence_history['loss'].append(total_loss.item())
    
            # Update tqdm description periodically
            if epoch % int(0.1*num_epochs) == 0 or epoch == num_epochs - 1:
                tqdm.write(f'Epoch {epoch:4d} | Recon: {recon_loss.item():.3f} | KL: {kl_loss.item():.3f} | Total: {total_loss.item():.3f}')
    
        self.vaeNet.encoder.isTraining = False
        checkpoint = {
            "model_state": self.vaeNet.state_dict(),
            "input_data": self.input_data,
            "scaling_info": self.scaling_info,
            "identifiers": self.identifiers,
            "architecture_config": self.architecture_config,
        }

        torch.save(checkpoint, saved_net_path)
        print(f"Training complete. Model saved to {saved_net_path}")
        return convergence_history
      
  
    
    # def iter_lossPlot(self,TrainLoss,valLoss,N):
      
    #     plt.figure(num=3,figsize=(8, 6))
    #     plt.plot(N,TrainLoss/TrainLoss[0],'r-o',label='Training loss',markersize=8)
    #     plt.plot(N,valLoss/TrainLoss[0],'b-s',label='Validation loss',markersize=8)
    #     # # Labels
    #     plt.ylabel('Loss',fontsize=16)
    #     plt.xlabel('Iteration',fontsize=16)
        
    #     # Title
    #     titleStr = f"Training Loss {TrainLoss[-1]:.2e}\nValidation Loss {valLoss[-1]:.2e}"
    #     plt.title(titleStr, fontsize=18)
    #     plt.legend(fontsize=16)
        # plt.show()
    
    def getProperties(self,data_normalized):
        # scaling_info = torch.stack((dataScaleMax,dataScaleMin,dataScaleMean,dataScaleStd),dim=1) format
    
        def unnorm(x, scaling_info):
          # x = (x +1.0) *0.5
          return ((x)*(scaling_info[0,:]-scaling_info[1,:]) + scaling_info[1,:])
          
        def unlogp(x):
          return torch.exp(x) - 1.0
          
        def unlog10(x):
          return (10**x) - 2.0
          
        def unzscore(x,scaling_info):
          return x *scaling_info[3,:] + scaling_info[2,:]
        
        data_denormalized = unnorm(data_normalized,self.scaling_info)

        return  data_denormalized

    def plot_latent_contour(self, save_file="latent_contour.png", save_figures=False,
                            fig=None):
        """
        Plot contour maps of decoded properties in 2D latent space.
        Accepts optional `fig`. Returns (fig, axes).
        """
      
        # ---- Compute latent points ----
        self.vaeNet.eval()
        with torch.no_grad():
            latent_points = self.vaeNet.encode(self.input_data).cpu().numpy()

        # Filter out any rows with non-finite values to avoid downstream
        # numerical/plotting issues (matplotlib can fail when axis limits
        # contain NaN/Inf values).
        finite_mask = np.isfinite(latent_points).all(axis=1)
        if not np.all(finite_mask):
            n_bad = np.count_nonzero(~finite_mask)
            print(f"Warning: {n_bad} latent points contained NaN/Inf and will be excluded from contour plotting.")
            latent_points = latent_points[finite_mask]

        if latent_points.shape[0] == 0:
            raise ValueError("All latent points are non-finite; cannot create latent contour plot.")
      
        latent_dim = self.architecture_config['decoder']['latentDim']
        if latent_dim != 2:
            raise ValueError("Contour plot requires latent_dim = 2.")
      
        # ---- Create grid in latent space ----
        x_min, x_max = latent_points[:, 0].min(), latent_points[:, 0].max()
        y_min, y_max = latent_points[:, 1].min(), latent_points[:, 1].max()
      
        xx_z, yy_z = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100)
        )

        grid_flat = np.column_stack((xx_z.ravel(), yy_z.ravel()))
        grid_tensor = torch.from_numpy(grid_flat).to(dtype=self.input_data.dtype, device=self.device) 
      
        # ---- Decode grid ----
        with torch.no_grad():
            decoded_grid = self.vaeNet.predict(grid_tensor).cpu()

        # Replace NaNs/Infs in decoder output before denormalization/reshaping
        decoded_grid = torch.nan_to_num(decoded_grid, nan=0.0, posinf=1e12, neginf=-1e12)

        decoded_grid = self.getProperties(decoded_grid)
        decoded_grid = decoded_grid.reshape(xx_z.shape[0], xx_z.shape[1], -1)

        # ---- Figure handling ----
        n_prop = decoded_grid.shape[2]

        # Determine square-ish rows and columns
        n_row = int(np.floor(np.sqrt(n_prop)))
        n_col = int(np.ceil(n_prop / n_row))
        
        # --- Dynamic figure size ---
        # base size per subplot (in inches)
        w_per_subplot = 4
        h_per_subplot = 3.5
        
        fig_width  = n_col * w_per_subplot
        fig_height = n_row * h_per_subplot
        
        # Create or reuse figure
        if fig is None:
            fig, axes = plt.subplots(n_row, n_col, figsize=(fig_width, fig_height))
        else:
            fig.clf()
            axes = fig.subplots(n_row, n_col)
            fig.set_size_inches(fig_width, fig_height)
            
        # ---- Plot properties ----
        flat_axes = axes.flatten()
        for i in range(decoded_grid.shape[2]):
            ax = flat_axes[i]
            cntr = ax.contour(xx_z, yy_z, decoded_grid[:, :, i], levels=20, cmap='jet')
            ax.clabel(cntr, inline=True, fontsize=8)
            ax.set_title(f"Property {i+1}")
            ax.set_xlabel("Latent Dim 1")
            ax.set_ylabel("Latent Dim 2")
            fig.colorbar(cntr, ax=ax)
      
        plt.tight_layout()
        if save_figures:
            plt.savefig(f"{save_file}")
        print(f"Saved contour plot → {save_file}")
      
        return fig, axes

  
    def plot_latent_scatter(self,
                            save_file="latent_scatter.png",
                            save_figures=False,
                            fig=None):
        """
        Scatter plot of latent space. Supports fig input.
        """
    
        # ---- Basic data ----
        labels = self.identifiers['classID']
        class_names = self.identifiers['Shape']
    
        colors = ['red','green','blue','orange','purple','cyan','pink','yellow','black','brown']
    
        self.vaeNet.eval()
        with torch.no_grad():
            latent_points = self.vaeNet.encode(self.input_data.float()).cpu().numpy()
    
        latent_dim = self.architecture_config['decoder']['latentDim']

        # Filter non-finite latent points to avoid plotting issues
        finite_mask = np.isfinite(latent_points).all(axis=1)
        if not np.all(finite_mask):
            n_bad = np.count_nonzero(~finite_mask)
            print(f"Warning: {n_bad} latent points contained NaN/Inf and will be excluded from scatter plotting.")
            latent_points = latent_points[finite_mask]
            # Filter labels so they align with latent_points
            labels = np.array(labels)[finite_mask]
    
        # ---- Figure handling ----
        if fig is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig.clf()
            ax = fig.subplots(1, 1)
    
        # ---- Plot each class cluster ----
        for cls in np.unique(labels):
            latent_xy = latent_points[labels == cls]
    
            # latent_xy = latent_subset[:, [dim_x, dim_y]]
    
            name = class_names[np.where(labels == cls)[0][0]]
    
            ax.scatter(latent_xy[:, 0], latent_xy[:, 1],
                       c=colors[cls % len(colors)], s=40, alpha=0.8, label=name)
    
            
    
        # ---- Decode for metrics ----
        with torch.no_grad():
            latent_tensor = torch.from_numpy(latent_points).to(device=self.device, dtype=self.input_data.dtype)
            decoded_latent = self.vaeNet.predict(latent_tensor).cpu()

        # Replace NaNs/Infs in decoded outputs before computing metrics/plots
        decoded_latent = torch.nan_to_num(decoded_latent, nan=0.0, posinf=1e12, neginf=-1e12)
        decoded_latent = self.getProperties(decoded_latent)
        decoded_input = self.getProperties(self.input_data.cpu())
    
        # ---- Print accuracy ----
        print("\nReconstruction Metrics:")
        for i in range(decoded_input.shape[1]):
            mae = mean_absolute_error(decoded_input[:, i], decoded_latent[:, i])
            rmse = root_mean_squared_error(decoded_input[:, i], decoded_latent[:, i])
            r2 = r2_score(decoded_input[:, i], decoded_latent[:, i])
            print(f"Property {i+1}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        
        fntLabel = 18;fontTick=16
        # ---- Plot formatting ----
        ax.set_xlabel("Latent Dim 1", fontsize=fntLabel)
        ax.set_ylabel("Latent Dim 2" if latent_dim != 1 else "Value", fontsize=fntLabel)
        # ax.set_title("Latent Space Scatter Plot")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(title="Classes")
        ax.tick_params(axis='both', which='major', labelsize=fontTick)
        plt.tight_layout()
        if save_figures:
            plt.savefig(f"{save_file}")
        print(f"Saved scatter plot → {save_file}")
    
        return fig, ax


    def plot_latent_scatter3D(self,
                            save_file="latent_scatter.png",
                            save_figures=False,
                            fig=None):
        """
        Scatter plot of latent space.
        Supports latent_dim = 2 or 3.
        """
    
        # ---- Basic data ----
        labels = np.array(self.identifiers['classID'])
        class_names = self.identifiers['Shape']
    
        colors = ['red','green','blue','orange','purple','cyan','pink','yellow','black','brown']
    
        self.vaeNet.eval()
        with torch.no_grad():
            latent_points = self.vaeNet.encode(self.input_data.float()).cpu().numpy()
    
        latent_dim = self.architecture_config['decoder']['latentDim']
    
        if latent_dim not in (2, 3):
            raise ValueError(f"Latent scatter only supports latent_dim=2 or 3, got {latent_dim}")
    
        # ---- Filter non-finite values ----
        finite_mask = np.isfinite(latent_points).all(axis=1)
        if not np.all(finite_mask):
            n_bad = np.count_nonzero(~finite_mask)
            print(f"Warning: {n_bad} latent points contained NaN/Inf and will be excluded.")
            latent_points = latent_points[finite_mask]
            labels = labels[finite_mask]
    
        # ---- Figure handling ----
        if fig is None:
            fig = plt.figure(figsize=(10, 8))
    
        fig.clf()
        if latent_dim == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
    
        # ---- Plot each class ----
        for cls in np.unique(labels):
            mask = labels == cls
            latent_cls = latent_points[mask]
            name = class_names[np.where(labels == cls)[0][0]]
    
            if latent_dim == 3:
                ax.scatter(latent_cls[:, 0],
                           latent_cls[:, 1],
                           latent_cls[:, 2],
                           c=colors[cls % len(colors)],
                           s=40, alpha=0.8, label=name) # type: ignore
            else:
                ax.scatter(latent_cls[:, 0],
                           latent_cls[:, 1],
                           c=colors[cls % len(colors)],
                           s=40, alpha=0.8, label=name) # type: ignore
        
        
        
        # ---- Decode for metrics ----
        with torch.no_grad():
            latent_tensor = torch.from_numpy(latent_points).to(device=self.device, dtype=self.input_data.dtype)
            decoded_latent = self.vaeNet.predict(latent_tensor).cpu()

        # Replace NaNs/Infs in decoded outputs before computing metrics/plots
        decoded_latent = torch.nan_to_num(decoded_latent, nan=0.0, posinf=1e12, neginf=-1e12)
        decoded_latent = self.getProperties(decoded_latent)
        decoded_input = self.getProperties(self.input_data.cpu())
    
        # ---- Print accuracy ----
        print("\nReconstruction Metrics:")
        for i in range(decoded_input.shape[1]):
            mae = mean_absolute_error(decoded_input[:, i], decoded_latent[:, i])
            rmse = root_mean_squared_error(decoded_input[:, i], decoded_latent[:, i])
            r2 = r2_score(decoded_input[:, i], decoded_latent[:, i])
            print(f"Property {i+1}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        
        fntLabel = 18;fontTick=16
        # ---- Plot formatting ----
        ax.set_xlabel("Latent Dim 1", fontsize=fntLabel)
        ax.set_ylabel("Latent Dim 2", fontsize=fntLabel)
        if latent_dim == 3:
            ax.set_zlabel("Latent Dim 3", fontsize=fntLabel) # type: ignore
    
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(title="Classes")
        ax.tick_params(axis='both', which='major', labelsize=fontTick)
    
        plt.tight_layout()
    
        if save_figures:
            plt.savefig(save_file)
    
        print(f"Saved scatter plot → {save_file}")
    
        return fig, ax

  
  
  
  
  
  # def plot_latent_space(self, latent_dim_1_idx, latent_dim_2_idx, plot_hull=True, annotate_head=False, save_file_name='latent_space_plot.png',saveClass=False):
  #   """
  #   Plots the 2D latent space projection.

  #   Args:
  #     latent_dim_1_idx (int): Index of the first latent dimension to plot (e.g., 0).
  #     latent_dim_2_idx (int): Index of the second latent dimension to plot (e.g., 1).
  #     plot_hull (bool): Whether to plot a convex hull around each class.
  #     annotate_head (bool): Whether to annotate the first few data points with their names.
  #     save_file_name (str): Name of the file to save the plot.

  #   Returns:
  #     tuple: Matplotlib figure and axes objects.
  #   """
  #   # Define a set of colors for different classes
  #   colors = ['red', 'green', 'orange', 'pink', 'blue', 'black', 'violet', 'cyan', 'purple', 'yellow']

  #   # Get class IDs and point labels from identifiers
  #   class_ids = self.identifiers['classID']
  #   # point_labels = self.identifiers['Name'] # Using 'Name' as the point label
  #   shapeName = self.identifiers['Shape']
  #   # Ensure the model is in evaluation mode before getting latent space
  #   self.vaeNet.eval()
  #   with torch.no_grad(): # Disable gradient calculation for inference
  #       # Pass input data through the encoder to get latent representations (z)
  #       _ = self.vaeNet.encoder(self.input_data) # This populates self.vaeNet.encoder.z
  #       z = self.vaeNet.encoder.z.to('cpu').numpy() # Move to CPU and convert to NumPy
  #   latentDim = self.architecture_config['decoder']['latentDim']

  #   # countour plot 
  #   if latentDim == 2:
  #     # Create a grid for contour plotting
  #     x_min, x_max = z[:, 0].min() , z[:, 0].max() 
  #     y_min, y_max = z[:, 1].min() , z[:, 1].max() 
  #     xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
      
  #     # Predict using the decoder for the grid points
  #     grid_points = np.column_stack((xx.ravel(), yy.ravel()))
  #     grid_tensor = torch.from_numpy(grid_points).float().to(self.device)
  #     AI_grid = self.vaeNet.decoder(grid_tensor).to('cpu').detach()
      
  #     # Get properties for the grid points
  #     AI_grid = self.getProperties(AI_grid)
  #     # print(AI_grid[:,-4].max().detach().numpy())
      
  #     # Reshape to match the grid shape
  #     AI_grid_reshaped = AI_grid.reshape(xx.shape[0], xx.shape[1], -1)
  #     AI_grid_reshaped = AI_grid_reshaped[:, :, -4::]  # Keep only the first three properties (Area, Ixx, Iyy)
  #     # Plot contour lines for each property in subplots
  #     fig2, axs_sub = plt.subplots(1, AI_grid_reshaped.shape[2], figsize=(15, 5),num=2) # add fig number
  #     for j in range(AI_grid_reshaped.shape[2]):
  #         # Also add countour line values
  #         cntr = axs_sub[j].contour(xx, yy, AI_grid_reshaped[:, :, j], levels=20, cmap='jet', alpha=0.8)
  #         axs_sub[j].clabel(cntr, inline=True, fontsize=8, fmt='%1.2f')  # Label contour lines
  #         # add colorbar
  #         axs_sub[j].set_title(f'Property {j+1} Contour Plot')
  #         axs_sub[j].set_xlabel('Latent Dimension 1')
  #         axs_sub[j].set_ylabel('Latent Dimension 2')
  #         axs_sub[j].scatter(z[:, 0], z[:, 1], c='cyan', s=10, alpha=0.2)  # Scatter plot of latent points
  #         # axs_sub[j].set_xlim([-2.,2.])
  #         # axs_sub[j].set_ylim([-0.0,2.0])
  #         fig2.colorbar(cntr, ax=axs_sub[j])

  #   plt.tight_layout()
  #   plt.savefig(save_file_name+'_prop.png')
    
  #   fig, axs = plt.subplots(num=1,figsize=(10, 8)) # Adjust figure size for better readability
  #   n=0
  #   # Iterate through each unique class ID
  #   for i in range(np.max(class_ids) + 1):
  #     # Filter latent points belonging to the current class
  #     z_class = z[class_ids == i, :]
  #     n = np.where((class_ids == i) == True)[0][0] # Get the index of the first occurrence of class i

  #     # Select the two latent dimensions for plotting
  #     if latentDim == 1:
  #       z_plot = np.vstack((z_class[:, latent_dim_1_idx], -i/10.0+0*z_class[:, latent_dim_1_idx])).T
  #     elif latentDim == 2:
  #       z_plot = np.vstack((z_class[:, latent_dim_1_idx], z_class[:, latent_dim_2_idx])).T
  #     # Scatter plot the points for the current class
  #     scatterplt = axs.scatter(z_plot[:, 0], z_plot[:, 1], c=colors[i % len(colors)], s=40, label=shapeName[n], alpha=0.8,edgecolors='none')
      
  #     # plt.pause(5.0)
  #     limit_min = -3
  #     limit_max =  2
     
  #     if saveClass:
  #       axs.set_xlim((limit_min, limit_max))
  #       axs.set_ylim((limit_min, limit_max)) # type: ignore
  #       plt.savefig(shapeName[n]+save_file_name)
  #       scatterplt.remove()

  #   AI = self.vaeNet.decoder(torch.from_numpy(z).to(self.device)).to('cpu').detach()
   
  #   # AI[:,-1] = 2*AI[:,-1]-AI[:,1]
    
  #   AIdata = self.input_data.to('cpu').detach()
    
    
  #   # print("ai_data",AIdata[0,:])
  #   # print("ai_pred",AI[0,:])
    
  #   AI = self.getProperties(AI) # prediction
  #   # AI = AI[:,-3::]
    
  #   AIdata= self.getProperties(AIdata) # denormalized data
  #   # AIdata[:,-1] = 2*AIdata[:,-1]-AIdata[:,1]
  #   # AIdata = AIdata[:,-3::]
    
    
  #   if latentDim == 1:
  #     # scatterplt = axs.scatter(z,AIdata[:,0], c='lightgreen', s=40, label='Area Data', alpha=0.5)
  #     scatterplt = axs.scatter(z,AI[:,-3], c='m', s=40, label='Area Prediction', alpha=0.5)
  #     plt.pause(5.0)
  #     # scatterplt = axs.scatter(z,AIdata[:,1], c='darkgreen', s=40, label='Ixx Data', alpha=0.5)
  #     scatterplt = axs.scatter(z,AI[:,-2], c='red', s=40, label='Ixx Prediction', alpha=0.5)
  #     plt.pause(5.0)
      
  #     # scatterplt = axs.scatter(z,AIdata[:,2], c='brown', s=40, label='Iyy Data', alpha=0.5)
  #     scatterplt = axs.scatter(z,AI[:,-1], c='cyan', s=40, label='Iyy Prediction', alpha=0.5)
  #     plt.pause(5.0)
    
  #   # print(z_class.shape)
  #   # AIdataI = self.input_data.to('cpu').detach()[class_ids == i,:]
  #   # AIn = self.vaeNet.decoder(torch.from_numpy(z_class.reshape((-1,1))).to(self.device)).to('cpu').detach()    
  #   # AIp = self.getProperties(AIn) # prediction
  #   # scatterplt = axs.scatter(z_class,AIp[:,1], c='blue', s=40, label='Ixx I', alpha=0.7)
  #   # scatterplt = axs.scatter(z_class,AIdataI[:,1], c='darkgreen', s=40, label='Ixx I Data', alpha=0.7)
  #   # scatterplt = axs.scatter(z_class,AIp[:,2], c='violet', s=40, label='Iyy I', alpha=0.7)
  #   # scatterplt = axs.scatter(z_class,AIdataI[:,2], c='brown', s=40, label='Iyy I Data', alpha=0.7)


  #   i=0 # 0 for area,
  #   for i in range(AIdata.shape[1]):
  #     # Calculate MAE using scikit-learn
  #     avg_mae = mean_absolute_error(AIdata[:,i], AI[:,i])
  
          
  #     # Calculate RMSE using scikit-learn
  #     avg_rmse = root_mean_squared_error(AIdata[:,i], AI[:,i])
  
  #     # Calculate R² score using scikit-learn
  #     r2_scores = r2_score(AIdata[:,i], AI[:,i], multioutput='uniform_average')
  #     print(f"Property {i+1}: MAE = {avg_mae:.4f}, RMSE = {avg_rmse:.4f}, R2 = {r2_scores:.4f}")
    
    
  #   # zall = torch.linspace(np.min(z),np.max(z),1000).reshape((-1,1)).to(self.device)
    
  #   # AI = self.vaeNet.decoder(zall).to('cpu').detach()
    
  #   # A,Ix,Iy = self.getProperties(AI)
    
  #   # scatterplt = axs.scatter(zall.to('cpu').detach(),A, c=colors[i % len(colors)], s=40, label='Area values', alpha=0.7)

  #   # Annotate individual points if requested
  #   # for i, txt in enumerate(point_labels):
  #   #   if annotate_head == False or (annotate_head == True and i < 26): # Keep original condition for now
  #   #     axs.annotate(txt, (z[i, latent_dim_1_idx], z[i, latent_dim_2_idx]), size=8, alpha=0.7)

    
  #   # activate first figure
  #   # plt.figure(fig.number)
  #   # axs = plt.gca()  # Get the current axes
  #   # Set plot labels and style
  #   plt.xlabel(f'Latent Dimension 1', size=14)
  #   plt.ylabel(f'Latent Dimension 2', size=14)
  #   if latentDim == 1:
  #       plt.ylabel(f'Properties values', size=14)

  #   plt.title('Latent Space Projection', size=16)
  #   plt.grid(True, linestyle='--', alpha=0.6)
  #   axs.spines['right'].set_visible(False)
  #   axs.spines['top'].set_visible(False)
  #   # axs.set_xlim([-2.,2.])
  #   # axs.set_ylim([-0.0,2.0])
  #   plt.legend(title='Classes')
  #   plt.tight_layout()
  #   plt.savefig(save_file_name+'.png')
  #   print(f"Latent space plot saved to {save_file_name}")
    
  #   if latentDim == 1: 
  #     print(np.min(z),np.max(z))
  #   if latentDim == 2:
  #     print(np.min(z[:, 0]), np.max(z[:, 0]))
  #     print(np.min(z[:, 1]), np.max(z[:, 1]))
  #   return fig, axs