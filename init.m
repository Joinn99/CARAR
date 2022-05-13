%% CARAR: Initialization script
clear;
if gpuDeviceCount > 0
    gpuDevice(1);
end
addpath('Model/', 'Utils/', 'Functions');
fprintf("Initlization complete.\n");