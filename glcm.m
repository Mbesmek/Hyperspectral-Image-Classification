function [features] = glcm(X)
[w h n]=size(X);
 offset=[0 1; -1 1; -1 0; -1 -1];

for i=1:n  
M = graycomatrix(X(:,:,i), 'NumLevels',256,'GrayLimits',[0,255],  'Offset', offset, 'Symmetric', false);
stats1 = graycoprops(M,'all');
features(:,i)= struct2array(stats1);


end