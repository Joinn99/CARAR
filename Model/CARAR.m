function [Md] = CARAR(Dt, Pm)
     % CARAR: Correaltion Aware Review Aspect Recommendation
     % ------------------------------------------------------------

     %% Hyperparameters
     Pm.r = 2;
     Pm.alpha = 1;
     % Pm.beta = 1;
     % Pm.phi = 0.1;

     %% Other settings
     Pm.pretrained = false;
     Pm.verbose = true;
     Pm.thres = 1e-7;
     Pm.type = 'gpuArray';

     %% Training
     [Pm.N, Pm.l] = size(Dt.E);
     [~, Pm.d] = size(Dt.D);
     [Pm.m, ~] = size(Dt.IS);
     [Pm.n, ~] = size(Dt.IU);  
     Md.Pm = Pm;
     Md = CARAR_C(Dt, Md, Md.Pm);
     Md = CARAR_LF(Dt, Md, Md.Pm);
     Md = CARAR_W(Dt, Md, Md.Pm);
end