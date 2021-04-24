function distance = euc(a,b) 
% Euclidean Distance 
%   calculates the Euclidean distance between two cases with an equal
%   number of features. 

% Author: Awabullah Syed 
% Date: 10th April 2021 

if nargin ~= 2 
    error('Two input arguments required.');
    return;
end 

if ~all(size(a) == size(b))
    error('Dimensions of inputs are not equalt.'); 
    return;
end 
if min(size(a)) ~= 1 
    error('Input is not a vector'); 
    return; 
end 

%Calculate the Euclidean Distance using the MATLAB's norm function 
distance = norm (a-b); 
end 