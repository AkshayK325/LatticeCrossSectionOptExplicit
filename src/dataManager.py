import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.interpolate import make_lsq_spline
# torch.set_default_dtype(torch.float32)



## 💾 III. Data Management
#-------------------------------------------------------------------------------

class DataManager:
    """
    A class to manage data generation, cleaning, saving, and loading for 
    B-spline cross-sectional shapes.
    """
    def __init__(self, geomProps):
        self.geomProps = geomProps
        self.fit_circle_cp = True
    
    def induced_control_points(self, shape_type='circle'):
        """
        Generates a set of 2N flattened control point coordinates [x0..x(N-1), y1..yN] 
        in the range [0, 1] for canonical shapes, where N is the number of 
        x-components and y-components (length 2N total).
        
        Parameters:
        - N (int): The number of x-components and y-components (e.g., N=4 for 8 total elements).
        - shape_type (str): One of ['circle', 'rectangle', 'ibeam', 'random'].
    
        Returns:
        - XY_flat (ndarray): Flattened control points (2N,).
        """
        N = self.geomProps.spline_geom.n_cp
        
        if N < 2:
            raise ValueError("N must be at least 2 for a meaningful B-spline section.")
            
        # Create normalized parameter space for generic point generation
        
    
        # --- 1. Define Canonical Shapes using the generic parameter u ---
        
        if shape_type == 'circle':
            
            if self.fit_circle_cp: # fit once
                u_fit  = np.linspace(0.0, 1.0, 50)
                x_fit = np.cos(u_fit * np.pi / 2.0)
                y_fit = np.sin(u_fit * np.pi / 2.0)
                knot_vec = self.geomProps.spline_geom.knot_vector
                x_vals = make_lsq_spline(u_fit,x_fit,t=knot_vec,k=self.geomProps.spline_geom.k)
                y_vals = make_lsq_spline(u_fit,y_fit,t=knot_vec,k=self.geomProps.spline_geom.k)
                
                x_vals = np.round(x_vals.c, decimals=3)
                y_vals = np.round(y_vals.c, decimals=3)
                self.x_circle_cp = x_vals
                self.y_circle_cp = y_vals
                self.fit_circle_cp = False
            else:
                x_vals = self.x_circle_cp
                y_vals = self.y_circle_cp
            
        elif shape_type == 'rectangle':
            # u is the normalized index (0 to 1) for defining the shape along the contour
            u = np.linspace(0.0, 1.0, N) 
            # Approximates a quarter square/rectangle
            pow_n = 4
            x_vals = 1.0 - (u**pow_n) 
            y_vals = 1.0 - np.flip(u**pow_n) 

        elif shape_type == 'I':
            # Approximates an I-beam by having most points near the axes (flange/web)
            v1 = np.random.uniform(0.25, 1.1)
            v2 = np.random.uniform(0.1, v1*0.2)
            # if np.random.randint(0, 2) == 0: # Vertical I-beam flange (large y, small x in the middle)
            x_vals = np.array([v2, v2, v1, v1, 0.0])
            y_vals = np.array([0.0,v2, v2, v1, v1])
            # else:
            #     x_vals = np.array([v1, v1, v2, v2, 0.0])
            #     y_vals = np.array([0.0,v1, v1, v2, v2])
                
            if N > 5:
                # Calculate distance between each consecutive point
                dists = np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2)
                cumulative_dist = np.concatenate(([0], np.cumsum(dists))) 
                total_len = cumulative_dist[-1]
                
                new_distances = np.linspace(0, total_len, N)
                
                x_vals = np.interp(new_distances, cumulative_dist, x_vals)
                y_vals = np.interp(new_distances, cumulative_dist, y_vals)

        
        elif shape_type == 'random':
            # For generalization, adding a completely random option is useful
            x_vals = np.random.uniform(0.1, 1.1, N)
            y_vals = np.random.uniform(0.1, 1.1, N)
         
        else:
            raise ValueError("Invalid shape_type. Choose from ['circle', 'rectangle', 'I', 'random'].")
    
        # --- 2. Add Noise and Scale (Generalization is independent of N) ---
        
        
        scale_factorx, scale_factory = 1.0,1.0
        
        # # Scale: Multiply by a uniform random number (0 to 1)
        if np.random.randint(0, 2) == 0:
            scale_factorx = np.random.uniform(0.5, 1.1)
            scale_factory = np.random.uniform(scale_factorx, 1.1)
        else:
            scale_factorx = np.random.uniform(0.5, 1.1)
            scale_factory = scale_factorx
        
        x_vals = x_vals * scale_factorx
        y_vals = y_vals * scale_factory
        
        # Add Noise: Add uniform noise (e.g., +/- 0.1)
        if np.random.randint(0, 2) == 0:
            noise_range = 0.01
            x_vals += max(x_vals)*np.random.uniform(-noise_range, noise_range, N)
            y_vals += max(y_vals)*np.random.uniform(-noise_range, noise_range, N)
    
        # Clip values to ensure they stay within the valid normalized range [0, 1]
        # The first and last points (on the axes) must be carefully preserved to maintain symmetry constraints.
        x_vals = np.clip(x_vals, 0.0, 1.0)
        y_vals = np.clip(y_vals, 0.0, 1.0)
        
        # Making sure Y values are bigger than X
        if shape_type != 'I':
            if max(x_vals) > max(y_vals):
                c = x_vals.copy()
                x_vals = np.flip(y_vals.copy())
                y_vals = np.flip(c.copy())
        
        # --- 3. Flatten and Return ---
        # Format: [x0, x1, ..., x(N-1), y1, y2, ..., yN]
        XY_flat = np.concatenate([x_vals[:-1], y_vals[1:]])
        
        return XY_flat
        
    def write_shape_data_to_file(self, output_filename, mode, shape_name, inputs, outputs, ID):
        """
        Writes the shape definition parameters (control points) and calculated 
        properties (metrics) to a formatted text file (CSV/TXT).
        
        Parameters:
        - params_array (ndarray): The 8 control point coordinates [x0..x3, y1..y4].
        - ai_results (ndarray): The 6 calculated metrics [O, S_f, A, J, Iy, Iz].
        """
        with open(output_filename, mode) as f:
            if mode == "w":
                # Header is dynamic based on the number of control points
                N = self.geomProps.spline_geom.n_cp - 1
                
                header_parts = [f"{'Shape':<25}", f"{'classID':<10}"]
                
                # Dynamic x and y coordinate headers
                x_headers = [f"x{i:<14}" for i in range(N)]
                y_headers = [f"y{i+1:<14}" for i in range(N)]
                
                header_parts.extend(x_headers)
                header_parts.extend(y_headers)
                if inputs.shape[0]%2 != 0: # odd number of inputs
                    header_parts.append(f"t{0:<14}")
                
                # Metrics headers
                metrics_headers = [
                    f"{'O':<12}", f"{'S_f':<15}", f"{'Area':<15}",
                    f"{'J':<15}", f"{'Iyy':<15}", f"{'Izz':<15}"
                ]
                header_parts.extend(metrics_headers)
                
                header = ", ".join(header_parts)
                f.write(header + "\n")  
            
            
            # Write one line of formatted data
            line = (
                f"{shape_name:<25}, {int(ID):<10}, "
                + ", ".join([f"{val:<15.10f}" for val in inputs]) + ", "
                + f"{outputs[0]:<12.1f}, "
                + ", ".join([f"{val:<15.10f}" for val in outputs[1:]])
            )  
            f.write(line + "\n")
            
            # Return a single row array for concatenation
            all_data_row = np.concatenate([inputs.reshape(1, -1), outputs.reshape(1, -1)], axis=1)
            return all_data_row

    def generate_and_clean_dataset(self, output_filename, CONTROL_POINT_MINMAX, n_samples, seedNum=1):
        """
        Generates a dataset of randomly perturbed B-spline shapes, calculates their 
        properties, and filters them based on validity (no self-intersections, 
        realistic area).
        """
      
        # Increase the initial number of samples to compensate for cleaning/filtering
        oversample_factor = 1 
        n_initial = int(oversample_factor * n_samples)
    
        all_data = []
        
        rng = np.random.default_rng(seedNum)
        
        current_mode = "w"
        nn = CONTROL_POINT_MINMAX.shape[1]  # Number of control point parameters
        # 1. Generate N initial random samples in the normalized [0, 1] space
        samples_normalized = rng.uniform(0, 1, (n_initial, nn)) # d*2-2 = 8 parameters
        
        # 2. Add canonical shapes (circle, rectangle, I-beam) with noise
        n_add = int(n_initial * 0.02)
        for i in range(n_add):
            # Add a mix of shape types
            i_cs = self.induced_control_points(shape_type='I').reshape(1, -1)
            circle_cs = self.induced_control_points(shape_type='circle').reshape(1, -1)
            rectangle_cs = self.induced_control_points(shape_type='rectangle').reshape(1, -1)
            if nn % 2 != 0: # odd number of inputs
                # add thickness parameter as well
                t_val = rng.uniform(0, 1, (1, 1))
                i_cs = np.hstack((i_cs, t_val))
                circle_cs = np.hstack((circle_cs, t_val))
                rectangle_cs = np.hstack((rectangle_cs, t_val))
            samples_normalized = np.vstack((samples_normalized, i_cs))
            samples_normalized = np.vstack((samples_normalized, circle_cs))
            samples_normalized = np.vstack((samples_normalized, rectangle_cs))

        # Normalize each row by its max value. np.max(..., axis=1) returns an
        # array of shape (n,), so repeat/tile it to (n, p) where p is the
        # number of columns so the division broadcasts correctly.
        row_max = np.max(samples_normalized, axis=1)
        if samples_normalized.ndim == 2:
            p = samples_normalized.shape[1]
            row_max = np.repeat(row_max[:, np.newaxis], p, axis=1)
        samples_normalized = samples_normalized / row_max
        
        
        # find if any two row rows are identical, if so add small noise to one of them
        _, unique_indices = np.unique(samples_normalized, axis=0, return_index=True)
        duplicate_indices = set(range(samples_normalized.shape[0])) - set(unique_indices)
        for idx in duplicate_indices:
            samples_normalized[idx] += rng.normal(scale=1e-6, size=samples_normalized.shape[1])
        
        # 3. Denormalize to the actual control point range [min, max]
        samples_denormalized = (samples_normalized * (CONTROL_POINT_MINMAX[1, :] - CONTROL_POINT_MINMAX[0, :]) + 
                                CONTROL_POINT_MINMAX[0, :])
        
        # flip the samples_denormalized such that 1st row becomes last row
        samples_denormalized = np.flipud(samples_denormalized)
        
        n_total = samples_denormalized.shape[0]
        
        ID_list = []
        
        iterBar =True
        loop = tqdm(range(n_total), desc="Generating data", ncols=75) if iterBar else range(n_total)
    
    
        # 4. Calculate properties and filter/clean the data
        for i in loop:
            XY_q = samples_denormalized[i, :].copy()
            
            # Calculate all 6 metrics
            prop = self.geomProps.evaluate_section_metrics(XY_q)
    
            # Filtering/Cleaning criteria: Area is non-zero/realistic (e.g., between 1 and 6)
            if prop[0] == 0.0 or np.abs(prop[2]) > 0.5:

                if prop[0] == 0.0:
                    shape_name = "Invalid_shape(O = 0)"  # Self-intersection detected
                    class_id = 0
                else:
                    shape_name = "Valid_shape(O = 1)"  # Valid shape
                    class_id = 1
                
                ID_list.append(class_id)
                
                data = self.write_shape_data_to_file(output_filename, current_mode, shape_name, XY_q, prop, class_id)
                all_data.append(data)
                current_mode = "a" # Switch to append mode after writing header
                
            else:
                ID_list.append(-1) # Mark as filtered out
            
            if len(all_data) == n_samples:
                print("\nReached the desired number of samples.")
                break
                
        print(f"Total data points generated: {len(all_data)}")
        
        print(f"Valid shapes: {ID_list.count(1)}, Invalid shapes: {ID_list.count(0)}")
        
        # 5. Plot a random subset of the valid shapes
        # if geoPlot and all_data:
        #     fig, axes = plt.subplots(num=1, nrows=5, ncols=5, figsize=(18, 12))
        #     axes_flat = axes.flatten()
            
        #     # Convert all_data to a numpy array for easier random indexing
        #     all_data_np = np.concatenate(all_data) 
        #     n_valid = all_data_np.shape[0]
            
        #     np.random.seed(seedNum)
            
        #     for id in range(axes_flat.shape[0]):
        #         # Randomly select a valid shape's parameters
        #         i = np.random.randint(0, n_valid)
        #         XY_q = all_data_np[i, 0:nn]
        #         prop = all_data_np[i, nn:]
        #         self.geomProps.spline_geom.plot_cross_section(XY_q, prop, ax=axes_flat[id])
        #     plt.savefig('geometries_BsplineFree_sym.png', bbox_inches='tight')
        
       
    
    
    def load_and_normalize_data(self, filePath, rmoveCols=[]):
        """
        Loads the processed data from the Excel file, separates control points 
        from metrics, and normalizes the metrics for VAE training.
        
        Returns:
        - trainingData (torch.Tensor): Normalized metrics for VAE input.
        - dataInfo (torch.Tensor): Min, Max, Mean, Std for un-normalization.
        - dataIdentifier (dict): Original shape labels and class IDs.
        - trainInfo (ndarray): Original (denormalized) metrics.
        """
        
        df = pd.read_csv(filePath, sep=',')
        
        # Data identifiers (Shape and classID)
        dataIdentifier = {
            'Shape': df[df.columns[0]],
            'classID': df[df.columns[1]]
        }
        
        # The VAE is trained on the full set of 8 control points + 6 metrics = 14 features
        # VAE features: [x0..x3, y1..y4, O, S_f, Area, J, Iy, Iz]
        
        # Input features (starts from the 2nd column after ID)
        # df.columns[2] is 'x0'
        data = df[df.columns[2::]].to_numpy()
        # remove col from reverse order to avoid index shift
        for col in sorted(rmoveCols, reverse=True):
            data = np.delete(data, col, axis=1)
        
        # Apply log scaling to moments of inertia (Iy, Iz) if needed
        normalizeWithLog = False
        normalizeWithLog1p = False 
        
        trainInfo = data.copy()
        
        if normalizeWithLog:
          # trainInfo[:, -2:] = np.log10(np.abs(data[:, -2:])) # Example Log transformation
          pass 
        elif normalizeWithLog1p:
          trainInfo[:, -2:] = np.log10(np.abs(data[:, -2:]) + 2.0) # Example Log1p
          pass
          
        # Calculate min and max for Min-Max scaling
        dataScaleMax = torch.tensor(np.max(trainInfo, axis=0))
        dataScaleMin = torch.tensor(np.min(trainInfo, axis=0))
        dataScaleMean = torch.tensor(np.mean(trainInfo, axis=0))
        dataScaleStd = torch.tensor(np.std(trainInfo, axis=0))
    
        # Normalize the data using Min-Max scaling to [0, 1]
        normalizedData = torch.tensor(trainInfo)
        normalizedData = (normalizedData - dataScaleMin) / (dataScaleMax - dataScaleMin)
        
        trainingData = normalizedData.clone()
        
        # Store scaling factors
        dataInfo = torch.stack((dataScaleMax, dataScaleMin, dataScaleMean, dataScaleStd), dim=0)
        
        return trainingData.float(), dataInfo.float(), dataIdentifier, trainInfo
    
    
    