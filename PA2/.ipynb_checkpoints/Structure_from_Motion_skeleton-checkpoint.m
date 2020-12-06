%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the skeleton code of PA2 in EC5301 Computer Vision.              %
% It will help you to implement the Structure-from-Motion method easily.   %
% Using this skeleton is recommended, but it's not necessary.              %
% You can freely modify it or you can implement your own program.          %
% If you have a question, please send me an email to haegonj@gist.ac.kr    %
%                                                      Prof. Hae-Gon Jeon  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear all;

addpath('Givenfunctions');

%% Define constants and parameters
% Constants ( need to be set )
number_of_iterations_for_5_point    = 0;

% Thresholds ( need to be set )
threshold_of_distance = 0; 

% Matrices
K               = [ 1698.873755 0.000000     971.7497705;
                    0.000000    1698.8796645 647.7488275;
                    0.000000    0.000000     1.000000 ];

%% Feature extraction and matching
% Load images and extract features and find correspondences.
% Fill num_Feature, Feature, Descriptor, num_Match and Match
% hints : use vl_sift to extract features and get the descriptors.
%        use vl_ubcmatch to find corresponding matches between two feature sets.



%% Initialization step
% Estimate E using 8,7-point algorithm or calibrated 5-point algorithm and RANSAC
E; % find out

% Decompose E into [R, T]
R; % find out
T; % find out

% Reconstruct 3D points using triangulation
X; % find out
X_with_color; % [6 x # of feature matrix] - XYZRGB

% Save 3D points to PLY
SavePLY('2_views.ply', X_with_color);
