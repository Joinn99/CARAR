%% CARAR: Demo script
% Author: Tianjun WEI (tjwei2-c@my.cityu.edu.hk)

init;                           % Initialization
Dataset = "OpenRiceEmoji";      % Dataset Name
GPU = true;                     % Use GPU
i=1;                            % Dataset fold (1-5)

% Load best hyperparameters or set it manually in Model/CARAR.m
run(sprintf("Params/%s.m", Dataset)) 

% Model Training
[TrainD, EvalD] = Load(Dataset, i - 1, GPU);
Model = CARAR(TrainD, Pm);

% Model Evaluation
fprintf("Dataset: %s\nFold: %d\nEvaluating...", Dataset, i);
Eval(Model, EvalD)
