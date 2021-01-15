function [rankFilter] =  MorpOpen(data);

f = data(:,:,:);

subplot(1,2,1);
imshow(data(:,:,100),[]);
[x,y,z]=size(f);

se = strel('arbitrary',3);
for k = 1:1:z
    f = data(:,:,k);
%     openData(:, :, k) = imopen(f, se); 
%     closeData(:, :, k) = imclose(openData(:, :, k), se);
%     topHat(:, :, k) = imtophat(f, se);
    rankFilter(:, :, k) = medfilt2(f);
%     mask = adapthisteq(f);
%     marker = imdilate(mask,se);
%     obr(:, :, k) = imreconstruct(marker,mask);
end



% % Read in image 
% img = imread('cameraman.tif'); 
%  
% % Original image is the mask 
% mask = img; 
%  
% % Create a marker image 
% marker = img - 1; 
%  
% % Structuring element for the dilation 
% % A 3x3 square implies 8-connectivity  
% se = strel('square', 3); 
%  
% % Perform morphological reconstruction 
% % i.e., geodesic dilation until stability 
% recon1 = marker; 
% recon1_old = zeros(size(recon1), 'uint8'); 
% while(sum(sum(recon1 - recon1_old)) ~= 0) 
%    % Retain output of previous iteration 
%    recon1_old = recon1; 
%     
%    % Perform dilation 
%    recon1 = imdilate(recon1, se); 
%     
%    % Restrict the dilated values using the mask 
%    bw = recon1 > mask; 
%    recon1(bw) = mask(bw);  
% end 
%  
% % Test against the output from the toolbox function 
% reconMatlab = imreconstruct(marker, img); 
% figure, imshow(reconMatlab - recon1, []), title('Difference image') 