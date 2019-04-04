function [rdiffpats, diffpats, probeOrig, positions] = ptychography_simulator(big_obj, sizeObj, sizeCCD, side, ratio)
% ========================================================================
%%% ptychography simulation (optical, x-ray)
% 1. initiate probe positions
% 2. create complex probes
% 3. couple each probe to diff patt of model image
% 
% INPUTS
% big_obj     big extended object
% sizeObj     size of extended object
% sizeCCD     size of CCD / probe / each position's exit wave
% side        1D scan size
% ratio       ratio of probe size to CCD array size
% 
% OUTPUTS
% rdiffpats     real space exit wave
% diffpats      propagated exit wave (i.e. FFT'ed)
% oROI          origion of region of interest
% probe_orig    simulated probe
% positions     scan positions
% 
% Author: Yuan Hung (Mike) Lo
% Email: lo.yuanhung@gmail.com
% Last edited: 20190404
% Jianwei (John) Miao Coherent Imaging Group
% University of California, Los Angeles
% Copyright (c) 2019. All Rights Reserved.

% ========================================================================
rng(2,'twister')

show_image = 0;
beam_stop = 0;
finite_CCD = 0;
finiteratio = 0;
diffpattnoise = 0;

% make probe positions (pixel size and real resolution need to be specified!)
positions = Grid_Scan(sizeObj, sizeCCD, side);
[Xmin, Ymin, Xmax, Ymax] = Find_Scan_Area(positions, sizeCCD);

numApertures = max(size(positions));

X = positions(:,1);
Y = positions(:,2);

figure(2) 
    scatter(positions(:,1), positions(:,2), 'bo'); grid on; grid minor; axis square; % check to see grid points
    xlim([0 sizeObj(2)])
    ylim([0 sizeObj(1)])
    title('Pixel scan positions')
    xlabel('Position (px)')
    ylabel('Position (px)')
    
%% make complex probe using positions
probeOrig = single(Make_Circle_Mask(round(ratio*sizeCCD(1)),sizeCCD(1)));

%% make individual probe data (apply finite CCD and beam stop effect )
diffpats = zeros(sizeCCD(1), sizeCCD(2),numApertures, 'single'); 
rdiffpats = zeros(sizeCCD(1), sizeCCD(2),numApertures, 'single'); 
diffpats_og = zeros(sizeCCD(1), sizeCCD(2),numApertures, 'single'); % used as buffer

for ii = 1:numApertures
    %ii
    objLocal = big_obj( Ymin(ii):Ymax(ii), Xmin(ii):Xmax(ii));
%     proj = probe_orig .* big_obj(oROI{ii,:}); % picks out r-space region of model object     
    exitWave = probeOrig .* objLocal; 
    rdiffpats(:,:,ii) = exitWave; % full diff patt to compare super resolution with 

    diffpats(:,:,ii) = abs(fft2(exitWave));  % keep it already fftshift_michaled and squarerooted 
    diffpats_og(:,:,ii) = diffpats(:,:,ii);  % record the origin pattern for comparison w noisy diff pats
    
    if show_image
%         showim(abs(probe_orig.*big_obj(oROI{ii,:}))); 
        showim(abs(exitWave)); pause(0.5)
    end
    
    if diffpattnoise ~= 0 % noisy
        diffpats(:,:,ii) = AddPNoise( abs(diffpats(:,:,ii)), diffpattnoise ) .* exp(1i*angle(diffpats(:,:,ii))); % noise free
    end
        
    if finite_CCD == 1 % super resolution
        diffpats(:,:,ii) = fftshift_michal(finiteCCD(fftshift_michal(diffpats(:,:,ii)),finiteratio*sizeCCD));
    end
    
    if beam_stop == 1 % missing center
        diffpats(:,:,ii) = fftshift_michal(beamstop(fftshift_michal(diffpats(:,:,ii)),stopsize)); % beamstop
    end
    
    if ii == 1 && show_image
        figure(271)
        imagesc(fftshift_michal(diffpats(:,:,1))),axis image
    end
end

end

    %% =======================================================================
    % HELPER FUNCTIONS
    % ========================================================================
    function positions = Grid_Scan(sizeObj, sizeCCD, side)
        extent = (sizeObj - sizeCCD)/3;
        xpos = linspace(-extent(1), extent(1), side);
        ypos = linspace(-extent(2), extent(2), side);
        [xpos_, ypos_] = meshgrid(xpos, ypos);

        % add random offset 
        random_offset = 1;
        if random_offset
            xpos_ = xpos_ + randn(size(xpos_)) * extent(1) / side / 3; % include variance in positions (normally distributed)
            ypos_ = ypos_ + randn(size(ypos_)) * extent(2) / side / 3;
        end

        xpos = round(xpos_(:) + sizeObj(2)/2+1);
        ypos = round(ypos_(:) + sizeObj(1)/2+1);
        positions = [xpos, ypos];
    end
    
    function out = Make_Circle_Mask(radius,imgSize)
        % Make a binary circle with defined radius
        nc = imgSize/2+1;
        n2 = nc-1;
        [xx, yy] = meshgrid(-n2:n2-1,-n2:n2-1);
        R = sqrt(xx.^2 + yy.^2);
        out = R<=radius;
    end

    function [Xmin, Ymin, Xmax, Ymax] = Find_Scan_Area(positions, sizeCCD)
        ycen = round(positions(:,2));
        xcen  = round(positions(:,1));
        Ymin = ycen - floor(sizeCCD(1)/2); 
        Ymax = Ymin + sizeCCD(1) - 1; % coordinates covered in each scan position
        Xmin = xcen - floor(sizeCCD(2)/2); 
        Xmax = Xmin + sizeCCD(2) - 1;
    end
