function Md = CARAR_W(Dt, Md, Pm)
    %% CARAR: Additional Information
    % Update parameters: W_D, W_S
    % W_D: R(d*l)        Additional information mapping
    % W_S: R(l*l)        Review aspect mapping 
    % l: Label number    d: Additional information dimension
    % ------------------------------------------------------------
    
    %% Functions
    sigm = @(x) (1+exp(-x)).^-1;
    EG = @(E, E_) E - sigm(E_);
    L1Prox = @(X, tau, G, eta) sign(X - G ./ max(tau, 1e-99)) .* max(abs(X - G ./ max(tau, 1e-99)) - eta ./ max(tau, 1e-99), 0);

    %% Fixed Parameters
    alphaW = 1;
    betaW = 1;

    %% Precalculation
    R = (eye(Pm.l, Pm.type) - Md.C) * (eye(Pm.l, Pm.type) - Md.C)';
    E_ = Md.Predict(Md, Dt, 0.0);
    DD = Dt.D' * Dt.D;
    DE = Dt.D' * Dt.E;
    DE_ = Dt.D' * E_;
    E_E = E_' * E_;
    %% Initialization
    if ~Pm.pretrained
        Md.WD.O = (DD + eye(Pm.d, Pm.type))^-1 * DE;
        Md.WD.P = Md.WD.O;
        Md.WD.tau = 0.25 .* sqrt(2) .* sqrt(sum(DD.^2, 'all') + sum((alphaW .* R).^2, 'all'));
        Md.WD.G = @(D, EG) -(D.D' * EG);
        Md.WS.O = (E_E + eye(Pm.l, Pm.type))^-1 * E_E;
        Md.WS.P = Md.WS.O;
        Md.WS.tau = 0.25 .* sum(E_E.^2, 1);
        Md.WS.G = @(D, EG) -(D.E' * EG);
        Md.sigm = sigm;
        Md.Adjust = @(Md, Dt, phi) (Dt.D * Md.WD.O + Md.Predict(Md, Dt, 0.0) * Md.WS.O) * (phi .* Md.C + (1 - phi) .* eye(Md.Pm.l));
    end

    %% Iteration Parameters
    k = 1;
    t = 1;
    gam = 0;   
    PreLoss = 1e9;
    
    %% Iteration 
    while k < 100
        %WD
        N = Md.WD.O + gam .* (Md.WD.O - Md.WD.P);
        ED = EG(Dt.E, Dt.D * N + E_ * Md.WS.O);
        Md.WD.P = Md.WD.O;
        Md.WD.O = L1Prox(N, Md.WD.tau, Md.WD.G(Dt, ED) - alphaW .* (N * R), betaW);
        % WE
        N = Md.WS.O + gam .* (Md.WS.O - Md.WS.P);
        ED = EG(Dt.E, Dt.D * Md.WD.O + E_ * N);
        Md.WS.P = Md.WS.O;
        Md.WS.O = L1Prox(N, Md.WS.tau, Md.WS.G(Dt, ED), betaW);
        % Loss
        ED = Dt.D * Md.WD.O + E_ * Md.WS.O;
        Loss = - sum(Dt.E .* log(1e-10 + sigm(ED)) +  (1 - Dt.E) .* log(1e-10 + 1 - sigm(ED)), 'all');
        %Loss = 0.5 .* sum((Dt.E - sigm(ED)).^2, 'all');
        Loss = Loss + 0.5 .* alphaW .* trace(R * Md.WD.O' * Md.WD.O) + betaW .* (norm(Md.WD.O, 1) + norm(Md.WS.O, 1));
        Loss = Loss ./ (Pm.N * Pm.l);
        % Evaluation
        Frac = abs(PreLoss - Loss) ./ PreLoss;
        if mod(k, 5) == 0 && Pm.verbose; fprintf("Iter: %d Loss: %.4f\t Frac: %.2e\n", k, Loss, Frac); end;
        if Frac < Pm.thres; break; end;
        % Epoch
        k = k + 1;
        t_1 = t;
        t = (1 + sqrt(1 + 4 * t^2)) / 2;
        gam = (t_1 - 1) / t;    
        PreLoss = Loss;
    end
end