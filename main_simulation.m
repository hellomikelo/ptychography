%{
Ptychography simulation script. Creates simulated data from chosen model
and reconstruct using ePIE.

Authors: Zhaoling Rong, Marcus Gallagher-Jones, Yuan Hung (Mike) Lo
Created: 2015
Last updated: 20190404
%}

clear
addpath('/u/home/y/y1lo/models')
cd('~/project-miao/ALS/ALS_COSMIC_Dec_2018_coral/ptychography/scripts/')

model = double(imread('lena.tif'));
model = model(256-64:256+63,256-64:256+63);
model = padarray(model,[64,64]);

side=10;
sizeObj = size(model);
sizeCCD = [164 164];
ratio = .2;
px_size = 1;

[rF2D, F2D, probe, positions] = ptychography_simulator(model, sizeObj, sizeCCD, side, ratio);
F2D=ifftshift(ifftshift(F2D,1),2);

disp('Data simulated!')

%% run reconstruction
ePIE_inputs(1).Patterns = F2D;
ePIE_inputs(1).Positions = positions;
ePIE_inputs(1).Iterations = 200;
ePIE_inputs(1).InitialObj = 0;
ePIE_inputs(1).InitialAp = probe;
% ePIE_inputs(1).FileName = sprintf('%s_complex',datestr(now,'yyyymmdd'));
ePIE_inputs(1).FileName = 'Simulation';
ePIE_inputs(1).PixelSize = px_size;
ePIE_inputs(1).GpuFlag = 0;
ePIE_inputs(1).ApRadius = 5;
ePIE_inputs(1).showim = 4;

% ePIE_inputs(1).model = big_obj;

ePIE(ePIE_inputs);
disp('Reconstruction complete!')