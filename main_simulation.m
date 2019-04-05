% Main ptychography simulation script. Generates simulated ptychography
% data and reconstruct using ePIE algorithm.
% 
% Author: Yuan Hung (Mike) Lo
% Email: lo.yuanhung@gmail.com
% Last edited: 20190404
% Jianwei (John) Miao Coherent Imaging Group
% University of California, Los Angeles
% Copyright (c) 2019. All Rights Reserved.

clear

%% generate simulated data
model = double(imread('lena.tif'));
model = model(256-64:256+63,256-64:256+63);
model = padarray(model,[64,64]);

%%% Tunable parameters
side = 10; % # of scan positions along one axis, e.g. side=10 means 10x10 scans
sizeObj = size(model); % size of the model
sizeCCD = [164 164]; % size of the detector
ratio = .2; % ratio of probe size to CCD array size, for defining the probe
px_size = 1; % pixel size of the reconstruction

[rF2D, F2D, probe, positions] = ptychography_simulator(model, sizeObj, sizeCCD, side, ratio);
F2D=ifftshift(ifftshift(F2D,1),2);

disp('Data simulated!')

%% run reconstruction
ePIE_inputs(1).Patterns = F2D;
ePIE_inputs(1).Positions = positions;
ePIE_inputs(1).Iterations = 200;
ePIE_inputs(1).InitialObj = 0;
ePIE_inputs(1).InitialAp = probe;
ePIE_inputs(1).FileName = 'Simulation';
ePIE_inputs(1).PixelSize = px_size;
ePIE_inputs(1).GpuFlag = 0;
ePIE_inputs(1).ApRadius = 5;
ePIE_inputs(1).showim = 4;

[best_obj, aperture, fourier_error, px_pos] = ePIE(ePIE_inputs);

disp('Reconstruction complete!')