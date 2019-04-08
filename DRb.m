%% sir-DR reconstruction algorithm

function [big_obj,aperture,fourier_error,initial_obj,initial_aperture] = DRb(ePIE_inputs,varargin)
%varargin = {beta_obj, beta_ap, probeNorm, init_weight, final_weight, order, semi_implicit_P}
rng('shuffle','twister');
%% setup working and save directories
dir = pwd;
save_string = [ dir '/Results_ptychography/']; % Place to save results
imout = 1; % boolean to determine whether to monitor progress or not

%% Load inputs from struct
gpu = ePIE_inputs(1).GpuFlag;
diffpats = ePIE_inputs(1).Patterns;
positions = ePIE_inputs(1).Positions;
pixel_size = ePIE_inputs(1).PixelSize;
big_obj = ePIE_inputs(1).InitialObj;
aperture_radius = ePIE_inputs(1).ApRadius;
aperture = ePIE_inputs(1).InitialAp;
iterations = ePIE_inputs(1).Iterations;
update_aperture = ePIE_inputs.updateAp;
showim = ePIE_inputs(1).showim;
if isfield(ePIE_inputs, 'do_posi'); do_posi = ePIE_inputs.do_posi;
else do_posi = 0; end
if isfield(ePIE_inputs, 'save_intermediate');
    save_intermediate = ePIE_inputs.save_intermediate;
else
    save_intermediate = 0;
end
%% === Reconstruction parameters frequently changed === %%
freeze_aperture = Inf;
optional_args = {.9, .1, 0.9, 1, 0.2, 0.6, 4, 0}; %default values for varargin parameters
nva = length(varargin);
optional_args(1:nva) = varargin;
[beta_obj, beta_ap, alpha, probeNorm, init_weight, final_weight, order, semi_implicit_P] = optional_args{:};

%% print parameters
fprintf('iterations = %d\n', iterations);
fprintf('gpu flag = %d\n', gpu);
fprintf('updating probe = %d\n', update_aperture);
fprintf('beta probe = %0.5f\n', beta_ap);
fprintf('beta object = %0.2f\n', beta_obj);
fprintf('probe norm = %d\n', probeNorm);
fprintf('Initial weight = %.2f, final weight = %.2f, order = %d\n',init_weight, final_weight, order)
fprintf('semi implicit on P = %d\n',semi_implicit_P);
clear ePIE_inputs
%% Define parameters from data and for reconstruction
for ii = 1:size(diffpats,3)
    diffpats(:,:,ii) = fftshift(sqrt(diffpats(:,:,ii)));
end
diffpats = single(diffpats);
[y_kspace,~] = size(diffpats(:,:,1)); % Size of diffraction patterns
[N1,N2,nApert] = size(diffpats);
little_area = y_kspace; % Region of reconstruction for placing back into full size image
%% Get centre positions for cropping (should be a 2 by n vector)
%positions = convert_to_pixel_positions(positions,pixel_size);
[pixelPositions, bigx, bigy] = ...
    convert_to_pixel_positions_testing5(positions,pixel_size,little_area);
centrey = round(pixelPositions(:,2));
centrex = round(pixelPositions(:,1));
centBig = round((bigx+1)/2);
Y1 = centrey - floor(y_kspace/2); Y2 = Y1+y_kspace-1;
X1 = centrex - floor(y_kspace/2); X2 = X1+y_kspace-1;
%% create initial aperture?and object guesses
if aperture == 0
    aperture = single(((2*makeCircleMask(round(aperture_radius./pixel_size),little_area))));
    aperture = my_ifft(aperture);
    initial_aperture = aperture;
else
    aperture = single(aperture);
    initial_aperture = aperture;
end
if big_obj == 0
    big_obj = single(rand(bigx,bigy)).*exp(1i*(rand(bigx,bigy)));
    %big_obj = zeros(bigx,bigy);
    initial_obj = big_obj;
else
    big_obj = single(big_obj);
    initial_obj = big_obj;
end
if save_intermediate == 1
    inter_obj = zeros([size(big_obj) floor(iterations/10)]);
    inter_frame = 0;
else inter_obj = [];
end

[XX,YY] = meshgrid(1:bigx,1:bigy);
X_cen = floor(bigx/2); Y_cen = floor(bigy/2);
R2 = (XX-X_cen).^2 + (YY-Y_cen).^2;
Kfilter = exp(-R2/(2*400)^2);
Kfilter = Kfilter/max(Kfilter(:));

fourier_error = zeros(iterations,nApert);
if strcmp(gpu,'GPU')
    display('========ePIE DR reconstructing with GPU========')
    diffpats = gpuArray(diffpats);
    fourier_error = gpuArray(fourier_error);
    big_obj = gpuArray(big_obj);
    aperture = gpuArray(aperture);
else
    display('========ePIE DR reconstructing with CPU========')
end
best_err = 100; % check to make sure saving reconstruction with best error
Z = diffpats.*exp(rand(N1,N2,nApert)); ds=1;
% user can choose the weight sequence how slowly it increases
% weight in (0,1)
% weight starts at a small value, such as 0.2 and then slowly increase to
% some larger value, such as 0.6
% the function can be continuouse or step function
% my default function is a step monomial function of order 4th, 
% initial value=0.2, final value =0.6
%images = zeros(380,380,iterations);
fprintf('Number of diff patterns = %d\n',size(diffpats,3));
%weights = init_weight + 0.5*(final_weight-init_weight)*round(2*((1:iterations)/iterations).^order);
weights = init_weight + (final_weight-init_weight)*((1:iterations)/iterations).^order;
%beta = sqrt((iterations-(1:iterations))/iterations); weights = init_weight*beta + (1-beta)*final_weight;

fprintf('big_obj size = %dx%d, aperture size = %dx%d, pixel_size = %f\n',size(big_obj), size(aperture),pixel_size);
%% Main ePIE itteration loop
disp('========beginning reconstruction=======');
for t = 1:iterations
    if t == 1, tic; end
    weight = weights(t);
    %if t==2, beta_ap = 0.001; end
    %{
    if t<200, weight=0.1;  %120,165,220,270,295
    elseif t<300, weight=0.2;
    elseif t<380, weight=0.3;
    elseif t<440, weight=0.4;
    elseif t<470, weight=0.5;
    elseif t<490, weight=0.6;
    else weight=0.7;
    end
    %}
    %% sequential
    for aper = randperm(nApert)
        %bigObjShifted = circshift(big_obj, [-1*(centrey(aper) - centBig) -1*(centrex(aper) - centBig)]);
        %u_old = croppedOut(bigObjShifted,y_kspace); % size 384x384
        u_old = big_obj(Y1(aper):Y2(aper), X1(aper):X2(aper));
        %object_max = max(abs(u_old(:))).^2;
        probe_max = max(abs(aperture(:))).^2;
%% Create new exitwave
        Pu_old = u_old.*(aperture);
        z_u = fft2(Pu_old);
        check_dp = abs(z_u);
        
        z = Z(:,:,aper);
        z_F = (1+alpha)*z_u - alpha*z;
        
        diffpat_i = diffpats(:,:,aper);
        missing_data = diffpat_i == -1; 
        k_fill = z_F(missing_data);
        
        % update z: tunning weight
        phase = angle(z_F);
        z_F =  (1-weight)*exp(1i*phase) .* diffpats(:,:,aper) + weight*z_F ;
        z_F(missing_data) = k_fill;
        z = z_F + alpha*(z- z_u);
        Z(:,:,aper)=z;        
        
%% Update the object
        Pu_new = ifft2(z);
        diff = Pu_old - Pu_new;
        dt = beta_obj/probe_max;
        %u_temp = (u_old + dt*Pu_new.*conj(aperture)) ./ ( 1 + dt*abs(aperture).^2 );
        u_temp = ( ((1-beta_obj)/dt)*u_old + Pu_new.*conj(aperture)) ./ ( (1-beta_obj)/dt + abs(aperture).^2 );
        
        if do_posi, u_new = max(0,real(u_new));
        else u_new = u_temp; end

        big_obj(Y1(aper):Y2(aper), X1(aper):X2(aper)) = u_new;
%% Update the probe
        if update_aperture == 1 
            object_max = max(abs(u_new(:))).^2;
            if t > iterations - freeze_aperture
                new_beta_ap = beta_ap*sqrt((iterations-t+1)/iterations);
            else
                new_beta_ap = beta_ap;
            end            
            ds = new_beta_ap./object_max;

            if semi_implicit_P            
                aperture = ((1-new_beta_ap)*aperture + ds*Pu_new.*conj(u_old)) ./ ( (1-new_beta_ap) + ds*abs(u_old).^2 );
            else 
                aperture = aperture - ds*conj(u_old).*(diff);
            end            
        end         
        fourier_error(t,aper) = sum(sum(abs(diffpat_i(missing_data ~= 1) - check_dp(missing_data ~= 1))))./sum(sum(diffpat_i(missing_data ~= 1)));
    end
    scale = max(max(abs(aperture)));
    if probeNorm == 1
        aperture = aperture/scale;
    end
    %{
    if t>iterations-100
    z = fftshift(fft2(big_obj)).*Kfilter;
    big_obj = ifft2(ifftshift(z));
    end
    %}
      %% show results
    if  mod(t,showim) == 0 && imout == 1;        
        figure(333)       
        %hsv_big_obj = make_hsv(big_obj,1);
        hsv_aper = make_hsv(aperture,1);
        subplot(2,2,1)
        imagesc(abs(big_obj)); axis image; colormap gray; title(['reconstruction pixel size = ' num2str(pixel_size)] )
        subplot(2,2,2)
        imagesc(hsv_aper); axis image; colormap gray; title('aperture single'); colorbar
        subplot(2,2,3)
        errors = sum(fourier_error,2)/nApert;
        fprintf('%d. Error = %f, scale = %.3f, dt = %.4f, ds = %.4f, weight = %.2f\n',t,errors(t),scale,dt, ds, weight);        
        plot(errors); ylim([0,0.5]);
        subplot(2,2,4)
        imagesc(log(fftshift(check_dp))); axis image
        drawnow     
        %figure(39);
        %img_i = abs(big_obj(276:555,276:555));
        %images(:,:,t) = img_i;
        %imagesc(img_i); axis image; colormap(jet); colorbar;
        %drawnow;
        %drawnow
        %writeVideo(writerObj,getframe(h)); % take screen shot        
    end
        
    mean_err = sum(fourier_error(t,:),2)/nApert;
    if best_err > mean_err
        best_obj = big_obj;
        best_err = mean_err;
    end         
    if save_intermediate == 1 && mod(t,10) == 0
        inter_frame = inter_frame + 1;
%         save(['inter_output_ePIE_ID_',jobID,'_itt_',num2str(itt),'.mat'],...
%         'best_obj','aperture','fourier_error','-v7.3');
        inter_obj(:,:,inter_frame) = best_obj;
    end
end
disp('======reconstruction finished=======')


drawnow

if strcmp(gpu,'GPU')
fourier_error = gather(fourier_error);
best_obj = gather(best_obj);
aperture = gather(aperture);
end


%save([save_string 'best_obj_' filename '.mat'],'best_obj','aperture','initial_obj','initial_aperture','fourier_error');


%% Function for converting positions from experimental geometry to pixel geometry

%     function [positions] = convert_to_pixel_positions(positions,pixel_size)
%         positions = positions./pixel_size;
%         positions(:,1) = (positions(:,1)-min(positions(:,1)));
%         positions(:,2) = (positions(:,2)-min(positions(:,2)));
%         positions(:,1) = (positions(:,1)-round(max(positions(:,1))/2));
%         positions(:,2) = (positions(:,2)-round(max(positions(:,2))/2));
%         positions = round(positions);
%         bigx =little_area + max(positions(:))*2+10; % Field of view for full object
%         bigy = little_area + max(positions(:))*2+10;
%         big_cent = floor(bigx/2)+1;
%         positions = positions+big_cent;
%         
%         
%     end
function [shiftDistancesX, shiftDistancesY, truePositions, positions, bigx, bigy] = ...
    convert_to_pixel_positions_testing3(positions,pixel_size,little_area)

        
        positions = positions./pixel_size;
        positions(:,1) = (positions(:,1)-min(positions(:,1)));
        positions(:,2) = (positions(:,2)-min(positions(:,2)));
        truePositions(:,1) = (positions(:,1) - max(positions(:,1))/2);
        truePositions(:,2) = (positions(:,2) - max(positions(:,2))/2);
        positions(:,1) = (positions(:,1)-round(max(positions(:,1))/2));
        positions(:,2) = (positions(:,2)-round(max(positions(:,2))/2));
        
        positions = round(positions);
        bigx =little_area + max(positions(:))*2+10; % Field of view for full object
        bigy = little_area + max(positions(:))*2+10;
%         bigx = little_area;
%         bigy = little_area;
        big_cent = floor(bigx/2)+1;
        positions = positions+big_cent;
        truePositions = truePositions + big_cent;
        
        shiftDistancesX = truePositions(:,1) - positions(:,1);
        shiftDistancesY = truePositions(:,2) - positions(:,2);
        
        
        
end

function [shiftDistancesX, shiftDistancesY, truePositions, positions, bigx, bigy] = ...
    convert_to_pixel_positions_testing4(positions,pixel_size,little_area)

        
        positions = positions./pixel_size;
        positions(:,1) = (positions(:,1)-min(positions(:,1)));
        positions(:,2) = (positions(:,2)-min(positions(:,2)));
        truePositions(:,1) = (positions(:,1) - round(max(positions(:,1))/2));
        truePositions(:,2) = (positions(:,2) - round(max(positions(:,2))/2));
        
        positions = round(truePositions);
        bigx =little_area + max(positions(:))*2+10; % Field of view for full object
        bigy = little_area + max(positions(:))*2+10;
%         bigx = little_area;
%         bigy = little_area;
        big_cent = floor(bigx/2)+1;
        positions = positions+big_cent;
        truePositions = truePositions + big_cent;
        
        shiftDistancesX = truePositions(:,1) - positions(:,1);
        shiftDistancesY = truePositions(:,2) - positions(:,2);
        
        
        
end

function [pixelPositions, bigx, bigy] = ...
    convert_to_pixel_positions_testing5(positions,pixel_size,little_area)

        
        pixelPositions = positions./pixel_size;
        pixelPositions(:,1) = (pixelPositions(:,1)-min(pixelPositions(:,1))); %x goes from 0 to max
        pixelPositions(:,2) = (pixelPositions(:,2)-min(pixelPositions(:,2))); %y goes from 0 to max
        pixelPositions(:,1) = (pixelPositions(:,1) - round(max(pixelPositions(:,1))/2)); %x is centrosymmetric around 0
        pixelPositions(:,2) = (pixelPositions(:,2) - round(max(pixelPositions(:,2))/2)); %y is centrosymmetric around 0
        
        bigx = little_area + round(max(pixelPositions(:)))*2+10; % Field of view for full object
        bigy = little_area + round(max(pixelPositions(:)))*2+10;

        big_cent = floor(bigx/2)+1;
        
        pixelPositions = pixelPositions + big_cent;
        
        
    end

%% 2D gaussian smoothing of an image

    function [smoothImg,cutoffRad]= smooth2d(img,resolutionCutoff)
        
        Rsize = size(img,1);
        Csize = size(img,2);
        Rcenter = round((Rsize+1)/2);
        Ccenter = round((Csize+1)/2);
        a=1:1:Rsize;
        b=1:1:Csize;
        [bb,aa]=meshgrid(b,a);
        sigma=(Rsize*resolutionCutoff)/(2*sqrt(2));
        kfilter=exp( -( ( ((sqrt((aa-Rcenter).^2+(bb-Ccenter).^2)).^2) ) ./ (2* sigma.^2) ));
        kfilter=kfilter/max(max(kfilter));
        kbinned = my_fft(img);
        
        kbinned = kbinned.*kfilter;
        smoothImg = my_ifft(kbinned);
        
        [Y, X] = ind2sub(size(img),find(kfilter<(exp(-1))));
        
        Y = Y-(size(img,2)/2);
        X = X-(size(img,2)/2);
        R = sqrt(Y.^2+X.^2);
        cutoffRad = ceil(min(abs(R)));
    end

%% Fresnel propogation
    function U = fresnel_advance (U0, dx, dy, z, lambda)
        % The function receives a field U0 at wavelength lambda
        % and returns the field U after distance z, using the Fresnel
        % approximation. dx, dy, are spatial resolution.
        
        k=2*pi/lambda;
        [ny, nx] = size(U0);
        
        Lx = dx * nx;
        Ly = dy * ny;
        
        dfx = 1./Lx;
        dfy = 1./Ly;
        
        u = ones(nx,1)*((1:nx)-nx/2)*dfx;
        v = ((1:ny)-ny/2)'*ones(1,ny)*dfy;
        
        O = my_fft(U0);
        
        H = exp(1i*k*z).*exp(-1i*pi*lambda*z*(u.^2+v.^2));
        
        U = my_ifft(O.*H);
    end

%% Make a circle of defined radius

    function out = makeCircleMask(radius,imgSize)
        
        
        nc = imgSize/2+1;
        n2 = nc-1;
        [xx, yy] = meshgrid(-n2:n2-1,-n2:n2-1);
        R = sqrt(xx.^2 + yy.^2);
        out = R<=radius;
    end

%% Function for creating HSV display objects for showing phase and magnitude
%  of a reconstruction simaultaneously

    function [hsv_obj] = make_hsv(initial_obj, factor)
        
        [sizey,sizex] = size(initial_obj);
        hue = angle(initial_obj);
        
        value = abs(initial_obj);
        hue = hue - min(hue(:));
        hue = (hue./max(hue(:)));
        value = (value./max(value(:))).*factor;
        hsv_obj(:,:,1) = hue;
        hsv_obj(:,:,3) = value;
        hsv_obj(:,:,2) = ones(sizey,sizex);
        hsv_obj = hsv2rgb(hsv_obj);
    end
%% Function for defining a specific region of an image

    function [roi] = get_roi(image, centrex,centrey,crop_size)
        
        bigy = size(image,1);
        bigx = size(image,2);
        
        half_crop_size = floor(crop_size/2);
        if mod(crop_size,2) == 0
            roi = {centrex - half_crop_size:centrex + (half_crop_size - 1);...
                centrey - half_crop_size:centrey + (half_crop_size - 1)};
            
        else
            roi = {centrex - half_crop_size:centrex + (half_crop_size);...
                centrey - half_crop_size:centrey + (half_crop_size)};
            
        end
    end

%% Fast Fourier transform function
    function kout = my_fft(img)
        kout = fftshift(fftn((ifftshift(img))));
    end
%% Inverse Fast Fourier transform function
    function realout = my_ifft(k)
        realout =fftshift((ifftn(ifftshift(k))));
        %realout =ifftshift((ifftn(fftshift(k))));
        
    end
end