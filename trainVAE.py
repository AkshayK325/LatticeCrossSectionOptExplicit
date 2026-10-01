
import sys
import os
sys.path.append(os.path.realpath('./src/'))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import time

import string
from vaeNet import VariationalAutoencoderModel

from splineGeom import SplineGeometry
from dataManager import DataManager
from geomProp import GeomProperties
# torch.set_default_dtype(torch.float32)

def plot_select_data_on_latent_space(splineNet, splineGeometry, data=None,z=None,fig=None):
    splineNet.vaeNet.to('cpu')
    
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        ax = fig.gca()
        plt.figure(fig.number)

    if data is not None:# input z ignored 
        z = splineNet.vaeNet.encode(data).to('cpu').detach() # Move to CPU
    elif z is None:
        raise ValueError("Either 'data' or 'z' must be provided.")

    for i in range(z.shape[0]):
        plt.scatter(z[i, 0], z[i, 1], s=100, facecolors='none', edgecolors='k',linewidths=2)  
        label = string.ascii_lowercase[i]  # 'a', 'b', 'c', ...
        plt.text(z[i, 0]+0.05, z[i, 1]+0.05, label, fontsize=16, color='black',fontweight='bold')# type:ignore
    
    decoded_data = splineNet.vaeNet.predict(z).to('cpu').detach()
    decoded_data_denormalized = splineNet.getProperties(decoded_data)

    #Fig num is + 1 of last fig used in this function
    existing_fignums = plt.get_fignums()
    next_fig_num = max(existing_fignums) + 1 if existing_fignums else 1
    _, axe = plt.subplots(num=next_fig_num,nrows=int(np.ceil(z.shape[0]/3)), ncols=3,figsize=(16,16))
    
    axes = axe.flatten() # Flatten the 2D array of axes for easy iteration
    if data is not None:
        data = splineNet.getProperties(data)
    else:
        data = decoded_data_denormalized.clone()
        
    XY = data[:,0:splineGeometry.n_cp*2-2]
    prop = data[:,splineGeometry.n_cp*2-2::]
    for i in range(z.shape[0]):
        splineGeometry.plot_cross_section(XY[i, :], prop[i, :], ax=axes[i], annotateCP=False, addLegend=False)
        label = string.ascii_lowercase[i]  # 'a', 'b', 'c', ...
        # plt.savefig(f'Geometry_{label}.png',bbox_inches='tight',dpi=300)
    plt.show(block=True)

#####################################################################################################
#####################################################################################################
# Design space parameters
n_cp = 5
k = 2
add_thickness = True
#####################################################################################################
n = 40000 # number of samples
saveToDisk = True
latentDim, numLayers, hiddenDim = 2, 2, 512 # 1x5x40
numLayers_nn, hiddenDim_nn = 2, 512
numEpochs = 20000
klFactor = 1e-7 #250 #100
learningRate = 1e-3
DeleteFiles = True
usePropertyPredictor = False

################################################################################################

SSminmax = np.zeros((2,n_cp*2-2)) 
SSminmax[0,:] = 0.0
SSminmax[1,:] = 1.2
if add_thickness:
    SSminmax = np.zeros((2,n_cp*2-2+1)) 
    SSminmax[0,0:-1] = 0.0
    SSminmax[1,0:-1] = 1.5
    SSminmax[0,-1] = 0.1
    SSminmax[1,-1] = 1.2
# SSminmax[1,2] = 2.0 #
splineGeometry = SplineGeometry(n_cp, k=k, npt=1000)
geomPropCal = GeomProperties(splineGeometry)
dataManagerAgent = DataManager(geomPropCal)

################################################################################################

# # testing code
# sample_num = 1
# # np.random.seed(1000)#1000 has pretty loop to test
# for seed_i in range(5):
#     np.random.seed(seed_i)
#     # XY_all = np.random.uniform(size=(sample_num,n_cp*2-2+1))#.reshape(-1)
#     XY_all = dataManagerAgent.induced_control_points(shape_type='circle').reshape(-1)
#     XY_all = np.concatenate([XY_all.reshape(1,-1), 0.2*np.ones((1,1))], axis=1) # add thickness
#     row_max = np.max(XY_all, axis=1)
#     XY_all = XY_all / row_max
#     XY_all = (XY_all * (SSminmax[1, :] - SSminmax[0, :]) + 
#                                     SSminmax[0, :])
#     prop = geomPropCal.evaluate_section_metrics(XY_all.reshape(-1))
#     splineGeometry.plot_cross_section(XY_all.reshape(-1), prop)
#     plt.show(block=True)
#     XY_all = dataManagerAgent.induced_control_points(shape_type='rectangle').reshape(-1)
#     XY_all = np.concatenate([XY_all.reshape(1,-1), 0.15*np.ones((1,1))], axis=1) # add thickness
#     row_max = np.max(XY_all, axis=1)
#     XY_all = XY_all / row_max
#     XY_all = (XY_all * (SSminmax[1, :] - SSminmax[0, :]) + 
#                                     SSminmax[0, :])
#     prop = geomPropCal.evaluate_section_metrics(XY_all.reshape(-1))
#     splineGeometry.plot_cross_section(XY_all.reshape(-1), prop)
#     plt.show(block=True)

# for i in range(1000):
    # np.random.seed(i)
    
    # XY_all = dataManagerAgent.induced_control_points(shape_type='rectangle').reshape(-1)
    # XY_all = (XY_all * (SSminmax[1, :] - SSminmax[0, :]) + 
    #                             SSminmax[0, :])
    # prop = dataManagerAgent.geomProps.evaluate_section_metrics(XY_all)
    # splineGeometry.plot_cross_section(XY_all, prop)
    # plt.show(block=True)
    
    # XY_all = dataManagerAgent.induced_control_points(shape_type='I').reshape(-1)
    # XY_all = (XY_all * (SSminmax[1, :] - SSminmax[0, :]) + 
    #                             SSminmax[0, :])
    # prop = dataManagerAgent.geomProps.evaluate_section_metrics(XY_all)
    # splineGeometry.plot_cross_section(XY_all, prop)
    # plt.show(block=True)
    
    # XY_all = dataManagerAgent.induced_control_points(shape_type='circle').reshape(-1)
    # XY_all = (XY_all * (SSminmax[1, :] - SSminmax[0, :]) + 
    #                             SSminmax[0, :])
    # prop = dataManagerAgent.geomProps.evaluate_section_metrics(XY_all)
    # axs = splineGeometry.plot_cross_section(XY_all, prop)
    
    # plt.show(block=True)
    
    # prop = dataManagerAgent.geomProps.evaluate_section_metrics(np.flip(XY_all))
    # splineGeometry.plot_cross_section(np.flip(XY_all), prop)
    # plt.show(block=True)
    # prop = dataManagerAgent.geomProps.evaluate_section_metrics(XY_all*1.1)
    # splineGeometry.plot_cross_section(XY_all*1.1, prop)
    # plt.show(block=True)

# XY_all = np.array([1.0, 1.0, 0.75,0.25,
#                     0.25,0.75, 1.0, 1.0]) # circle
# XY_all = np.array([1.0, 1.0, 1.0,0.5,
#                 0.5,1.0,1.0, 1.0]) # square          
# XY_all = np.array([1.0, 1.0, 1.0,0.5,
#                 0.5,1.0,1.0, 1.0])*[1,1,1,1,0.5,0.5,0.5,0.5]
# XY_all = np.array([1.38327707, 1.30820967, 0.17350359, 1.23384364, 0.77354709, 0.25623757,
#  0.73963629, 1.1671683 ])
# XY_all = np.array([1.0, 1.25, 0.1,0.8,
#                     0.8,0.1, 1.25, 1.0])# 4 sided clover

################################################################################################

addName = str(int(n/1000))+'k'+'_n_cp_'+str(n_cp) # change this to '' when not using symm
output_filename = "geomData/geometric_data_"+addName+".txt" # (2 for area, Ixx>=Iyy)

if not os.path.exists(output_filename):
    saveToDisk = True

start = time.perf_counter()
if saveToDisk :
    dataManagerAgent.generate_and_clean_dataset(output_filename, SSminmax, n, seedNum=1)
# sys.exit()
# [O , S_f , Area , J , Iyy , Izz]
trainingData, dataInfo, dataIdentifier, trainInfo = \
dataManagerAgent.load_and_normalize_data(output_filename,rmoveCols=[]) # type:ignore
elapsed_seconds = time.perf_counter() - start
hours, rem = divmod(elapsed_seconds, 3600)
minutes, seconds = divmod(rem, 60)
print(f'data gen and cleaning time : {int(hours)} hours, {int(minutes)} min, {int(seconds)} sec')

_, numFeatures = trainingData.shape

savedNet = 'savedNet/splineNet'+addName+'_'+str(numLayers)+'x'+str(hiddenDim)+'nfeatures_'+str(numFeatures)+'_zdim_'+str(latentDim)
save_latent = 'ResultsFig/latent_space_'+addName+'_'+str(numLayers)+'x'+str(hiddenDim)+'nfeatures_'+str(numFeatures)+'_zdim_'+str(latentDim)
save_contour = 'ResultsFig/latent_space_contour_'+addName+'_'+str(numLayers)+'x'+str(hiddenDim)+'nfeatures_'+str(numFeatures)+'_zdim_'+str(latentDim)

if usePropertyPredictor:
    # Geometry = control points (but excluding 2 fixed ones)
    geometry_dim = n_cp * 2 - 2 + 1 # include outcome O

    # Total number of features stored in the dataset:
    # geometry_dim + property_dim = numFeatures
    property_dim = numFeatures - geometry_dim

    encoder_input_dim = geometry_dim               # VAE always encodes the whole feature vector
    decoder_output_dim = geometry_dim               # Decoder reconstructs ONLY geometry
    predictor_output_dim = property_dim             # Predictor predicts ONLY properties

else:
    # No separate property predictor:
    encoder_input_dim = numFeatures
    decoder_output_dim = numFeatures              # Decoder reconstructs everything
    predictor_output_dim = 0                      # No predictor used

architecture_config = {
    'encoder': {
        'inputDim': encoder_input_dim,
        'hiddenDim': hiddenDim,
        'latentDim': latentDim,
        'numLayers': numLayers
    },
    'decoder': {
        'latentDim': latentDim,
        'hiddenDim': hiddenDim,
        'outputDim': decoder_output_dim,
        'numLayers': numLayers
    },
    'predictor': None
}
if usePropertyPredictor:
    architecture_config['predictor'] = {
        'inputDim': latentDim,
        'hiddenDim': hiddenDim_nn,
        'outputDim': predictor_output_dim,
        'numLayers': numLayers_nn,
        'dropout': 0.0
    }

print(architecture_config)
if DeleteFiles and os.path.exists(savedNet+'.pth'):
    os.remove(savedNet+'.pth')
    print('Deleted the exisiting pth file:',savedNet+'.pth')
 
start = time.perf_counter()
if not os.path.exists(savedNet+'.pth'):
    
    splineNet = VariationalAutoencoderModel(trainingData, dataInfo, 
                                         dataIdentifier, architecture_config,useCPU=False)
    for i in range(1):
        convgHistory = splineNet.train_model(numEpochs, klFactor, savedNet+'.pth', learningRate)
    fig,axs = splineNet.plot_latent_scatter3D(save_file=save_latent,save_figures=True)
    elapsed_seconds = time.perf_counter() - start
    minutes, seconds = divmod(elapsed_seconds, 60)
    print(f'training time : {int(minutes)} min, {int(seconds)} sec')

else: 
    checkpoint = torch.load(savedNet+'.pth')
    splineNet = VariationalAutoencoderModel(checkpoint['input_data'],checkpoint['scaling_info'],checkpoint['identifiers'],checkpoint['architecture_config'],useCPU=True)
    splineNet.load_model_from_file(checkpoint['model_state'])    
    print('model loading time : {:.2f} sec'.format(time.perf_counter() - start))
    fig,axs = splineNet.plot_latent_scatter3D(save_file=save_latent,save_figures=True)

# splineNet.plot_latent_contour(save_file=save_contour,save_figures=True)
plt.show(block=True)


dataSelect = trainingData[[-50,-200,-199],:]
ds_1 = trainingData[(trainingData[:, 8] == 0) , :]
ds_1 = ds_1[(ds_1[:, 10] < 0.5), :]

ds_2 = trainingData[(trainingData[:, 8] == 1) , :]
ds_2 = ds_2[(ds_2[:, 10] < 0.5), :]

dataSelect = torch.concatenate([dataSelect, ds_1[[0,int(ds_1.shape[0]/2),-1], :], ds_2[[0,int(ds_2.shape[0]/2),-1], :]], axis=0) # type: ignore

# plot_select_data_on_latent_space(splineNet, splineGeometry, data=dataSelect, fig=fig)