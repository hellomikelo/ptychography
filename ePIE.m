function [best_obj, aperture, fourier_error, initial_obj, initial_aperture, px_pos] = ePIE(ePIE_inputs,varargin)
% ePIE algorithm for ptychography phase retrieval.
% 
% Reference: Maiden & Rodenburg, Ultramicroscopy (2009) 
% 
% Source: Arjun Rana's Github page: https://github.com/arana89/ptycho
% Author: Yuan Hung (Mike) Lo
% Email: lo.yuanhung@gmail.com
% Last edited: 20190404
% Jianwei (John) Miao Coherent Imaging Group
% University of California, Los Angeles
% Copyright (c) 2019. All Rights Reserved.

%% =======================================================================
% INITIALIZATION
% ========================================================================
rng('shuffle','twister');
%varargin = {beta_obj, beta_ap}
optional_args = {0.9 0.9}; %default values for varargin parameters
nva = length(varargin);
% optional_args(1:nva) = varargin;
[beta_obj, beta_ap] = optional_args{:};
freeze_aperture = 10; % # of iterations to fix the probe before updating it
best_err = 1e6; % check to make sure saving reconstruction with best error

%% Unpack inputs from ePIE_inputs struct
iterations      = ePIE_inputs.Iterations;
px_size         = ePIE_inputs.PixelSize;
diff_pats       = ePIE_inputs.Patterns;
    % preprocess the patterns
    for ii = 1:size(diff_pats,3), diff_pats(:,:,ii) = single(fftshift(diff_pats(:,:,ii))); end
    [little_area, ~, num_apertures] = size(diff_pats); % Size of diffraction patterns
    fourier_error = zeros(iterations, num_apertures);
positions       = ePIE_inputs.Positions;
    % get the coordinates for cropping out each scan area
    px_pos = Convert_to_Pixel_Positions(positions, px_size);
    [Xmin, Ymin, Xmax, Ymax] = Find_Scan_Area(px_pos);
big_obj         = ePIE_inputs.InitialObj;
aperture_radius = ePIE_inputs.ApRadius;
aperture        = ePIE_inputs.InitialAp;
filename        = ePIE_inputs.FileName;

if isfield(ePIE_inputs, 'updateAp'), update_aperture = ePIE_inputs.updateAp; else update_aperture = 1; end
if isfield(ePIE_inputs, 'GpuFlag'), gpu = ePIE_inputs(1).GpuFlag; else gpu = 0; end
if isfield(ePIE_inputs, 'miscNotes'), miscNotes = ePIE_inputs.miscNotes; else miscNotes = 'None'; end
if isfield(ePIE_inputs, 'showim'), showim = ePIE_inputs(1).showim; else showim = 0; end
if isfield(ePIE_inputs, 'do_posi'), do_posi = ePIE_inputs.do_posi; else do_posi = 0; end
if isfield(ePIE_inputs, 'save_intermediate'); save_intermediate = ePIE_inputs.save_intermediate; else save_intermediate = 0; end
clear ePIE_inputs

%% Initialization
% initialize big object
if big_obj == 0
    big_obj = single(rand(bigx,bigy)).*exp(1i*(rand(bigx,bigy))) *1e2;
    initial_obj = big_obj;    
else
    big_obj = single(big_obj);
    initial_obj = big_obj;
end
best_obj = big_obj;

% initialize probe
if aperture == 0
    aperture = single(Make_Circle_Mask(round(aperture_radius./px_size),little_area));
    initial_aperture = aperture;    
else
    aperture = single(aperture); 
    initial_aperture = aperture;    
end

if gpu == 1
    diff_pats = gpuArray(diff_pats);
    fourier_error = gpuArray(fourier_error);
    big_obj = gpuArray(big_obj);
    aperture = gpuArray(aperture);
end

%% =======================================================================
% MAIN EPIE RECONSTRUCTION
% ========================================================================
disp('======== Parameters')
fprintf('Total iterations: %d\n', iterations);
fprintf('Beta probe: %0.1f\n', beta_ap);
fprintf('Beta object: %0.1f\n', beta_obj);
fprintf('GPU flag: %d\n', gpu);
fprintf('Updating probe: %d\n', update_aperture);
fprintf('Positivity: %d\n', do_posi);
fprintf('Misc notes: %s\n', miscNotes);
disp('======== ePIE: Reconstruction started');
for itt = 1:iterations
    for aper = randperm(num_apertures)
        %%% Get local scan area
        rspace = big_obj(Ymin(aper):Ymax(aper), Xmin(aper):Xmax(aper));
        buffer_rspace = rspace;
        object_max = max(abs(rspace(:))).^2;
        probe_max = max(abs(aperture(:))).^2;
        
        %%% Create new exit wave
        buffer_exit_wave = rspace.*(aperture);
        update_exit_wave = buffer_exit_wave;
        temp_dp = fft2(update_exit_wave);
        check_dp = abs(temp_dp);
        current_dp = diff_pats(:,:,aper);
        missing_data = current_dp == -1;
        k_fill = temp_dp(missing_data);
        temp_dp = current_dp.*exp(1i*angle(temp_dp)); % Replace reconstructed magnitudes with measured magnitudes
        temp_dp(missing_data) = k_fill;
        
        %%% Update object
        temp_rspace = ifft2(temp_dp);
        new_exit_wave = temp_rspace;
        diff_exit_wave = new_exit_wave-buffer_exit_wave;
        update_factor_ob = beta_obj/probe_max;
        new_rspace = buffer_rspace + update_factor_ob.*conj(aperture).*(diff_exit_wave);
        if do_posi == 1
            new_rspace = max(0, real(new_rspace));
        end
        big_obj(Ymin(aper):Ymax(aper), Xmin(aper):Xmax(aper)) = new_rspace;

        %%% Update probe
        if update_aperture == 1
            if itt > iterations - freeze_aperture
                new_beta_ap = beta_ap*sqrt((iterations-itt)/iterations);
                update_factor_pr = new_beta_ap./object_max;
            else
                update_factor_pr = beta_ap./object_max;
            end
            aperture = aperture +update_factor_pr*conj(buffer_rspace).*(diff_exit_wave);
        end
        
        fourier_error(itt,aper) = sum(sum(abs(current_dp(missing_data ~= 1) - check_dp(missing_data ~= 1))))./sum(sum(current_dp(missing_data ~= 1)));
    end
    
    %%% Calculate k-space error and update best object
    mean_err = sum(fourier_error(itt,:),2)/num_apertures;
    if mean_err < best_err
        best_obj = big_obj;
        best_err = mean_err;
    end         
    
    %%% Display intermediate results
    if  mod(itt,showim) == 0 && showim ~= 0
        errors = sum(fourier_error,2)/num_apertures;
        fprintf('%d. Error = %f, max(probe) = %f\n', itt, errors(itt), max(max(abs(aperture))));        
        
        figure(3); set(gcf, 'color', 'w', 'position', [95 621 1383 428]); colormap gray
        subplot(1,6,1:2)
            imagesc(abs(best_obj)); axis image off; colorbar
            title(['Object, itr ' num2str(itt)])
%             title(['reconstruction pixel size = ' num2str(pixel_size)] )
        subplot(1,6,3:4)
            imagesc((abs(aperture))); axis image off; colorbar
            title('Probe');
        subplot(1,6,5)
            plot(errors, 'linewidth', 2); axis square % ylim([0,0.2]); 
            title(sprintf('Avg Fourier error\n(%.5f)', errors(itt)))
        subplot(1,6,6)
            imagesc(log(fftshift(check_dp))); axis image off
            title('K-space')
        drawnow 
        
        %%% Save intermediate progress
        if 1; 
            export_fig(['../results/' filename '.png']); 
            save(['../results/' filename '_checkpoint.mat'], 'itt', 'best_obj','aperture','fourier_error','-v7.3'); 
        end
    end    
end
disp('====== ePIE: Reconstruction finished')

if gpu == 1
    fourier_error = gather(fourier_error);
    best_obj = gather(best_obj);
    aperture = gather(aperture);
end

% if saveOutput == 1
%     save([save_string 'best_obj_' filename '.mat'],'best_obj','aperture','initial_obj','initial_aperture','fourier_error');
% end

    %% =======================================================================
    % HELPER FUNCTIONS
    % ========================================================================
    function [positions] = Convert_to_Pixel_Positions(positions, pixel_size)
        % Convert experimental scan coordinates into pixel units
        positions = positions./pixel_size;
        positions(:,1) = (positions(:,1)-min(positions(:,1)));
        positions(:,2) = (positions(:,2)-min(positions(:,2)));
        positions(:,1) = (positions(:,1)-round(max(positions(:,1))/2));
        positions(:,2) = (positions(:,2)-round(max(positions(:,2))/2));
        positions = round(positions);
        bigx = little_area + max(positions(:))*2+10; % Field of view for full object
        bigy = little_area + max(positions(:))*2+10;
        big_cent = floor(bigx/2)+1;
        positions = positions+big_cent;
    end

    function [Xmin, Ymin, Xmax, Ymax] = Find_Scan_Area(pxPos)
        % Get the pixel extents of scan area: xmin, xmax, ymin, max
        yScanCens= round(pxPos(:,2));
        xScanCens = round(pxPos(:,1));
        % Define coordinates covering the extent of each scan position
        Ymin = yScanCens - floor(little_area/2); 
        Ymax = Ymin + little_area-1; 
        Xmin = xScanCens - floor(little_area/2); 
        Xmax = Xmin + little_area-1;
    end

    function out = Make_Circle_Mask(radius,imgSize)
        % Make a binary circle with defined radius
        nc = imgSize/2+1;
        n2 = nc-1;
        [xx, yy] = meshgrid(-n2:n2-1,-n2:n2-1);
        R = sqrt(xx.^2 + yy.^2);
        out = R<=radius;
    end

end



