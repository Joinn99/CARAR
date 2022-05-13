function [Md] = CARAR_LF(Dt, Md, Pm)
    % CARAR: Latent Factors Updating
    % Update parameters: b^G, b^I, b^U, p^I, p^U, Q^I, Q^U
    % b^G: R(l)                 Global bias
    % b^I: R(m*l)               Item sparse bias
    % b^U: R(n*l)               User sparse bias
    % p^I: R(m*r)               Item embedding
    % p^U: R(n*r)               User embedding
    % Q^I: R(r*l)               Review aspect embedding corresponding to users
    % Q^U: R(r*l)               Review aspect embedding corresponding to items
    % m: Item number            n: User number
    % l: Label number           r: Latent factor dimension
    % ------------------------------------------------------------
        
    %% Functions
        UI = @(X) X(Dt.iu, :, :);
        UR = @(X) pmul_p(Dt.IU, X);
        SI = @(X) X(Dt.is, :, :);
        SR = @(X) pmul_p(Dt.IS, X);
        sigm = @(x) (1+exp(-x)) .^ -1;
        EG = @(C, E, E_) (E - sigm(E_)) + (E - sigm(E_ * C)) * C';
    %% Proximal Operator
        L1Prox = @(X, tau, G, eta) sign(X - G ./ (tau + 1e-99)) .* max(abs(X - G ./ (tau + 1e-99)) - eta ./ (tau + 1e-99), 0);
        L2Prox = @(X, tau, G, eta) (tau ./ (tau + eta)) .* (X - G ./ (tau + 1e-99));
    %% Precomputation
        Md.Coef = 0.25 .* ones(1, Pm.l) .* (1 + norm(Md.C, 2)^2);
    %% Initialization
        if ~Pm.pretrained
            Md.B.O = 0.5 + 0 .* ones(1, Pm.l, Pm.type);
            Md.B.P = Md.B.O;
            Md.B.tau = @(M, D) M.Coef .* Pm.N;
            Md.B.G = @(M, D, ED) -sum(ED, 1);

            Md.AU.O = 0.0 .* ones(Pm.n, Pm.l, Pm.type);
            Md.AU.P = Md.AU.O;
            Md.AU.tau = @(M, D) sum(D.IU, 2) * M.Coef; 
            Md.AU.G = @(M, D, ED) -UR(ED);

            Md.AS.O = 0.0 .* ones(Pm.m, Pm.l, Pm.type);
            Md.AS.P = Md.AS.O;
            Md.AS.tau = @(M, D) sum(D.IS, 2) * M.Coef; 
            Md.AS.G = @(M, D, ED) -SR(ED);

            Md.EU.O = 0.1 .* randn(Pm.r, Pm.l, Pm.type);
            Md.EU.P = Md.EU.O;
            Md.EU.tau = @(M, D) M.Coef .* sum(UI(M.RU.O.^2), 'all');
            Md.EU.G = @(M, D, ED) -sum(UI(M.RU.O)' * ED, 1);

            Md.ES.O = 0.1 .* randn(Pm.r, Pm.l, Pm.type);
            Md.ES.P = Md.ES.O;
            Md.ES.tau = @(M, D) M.Coef .* sum(SI(M.RS.O.^2), 'all');
            Md.ES.G = @(M, D, ED) -sum(SI(M.RS.O)' * ED, 1);

            Md.RU.O = 0.1 .* randn(Pm.n, Pm.r, Pm.type);
            Md.RU.P = Md.RU.O;
            Md.RU.tau = @(M, D) sum(M.Coef ./ Pm.l, 2) .* full(sum(D.IU, 2)) .* sum(M.EU.O.^2, 'all');
            Md.RU.G = @(M, D, ED) -UR(ED * M.EU.O');

            Md.RS.O = 0.1 .* randn(Pm.m, Pm.r, Pm.type);
            Md.RS.P = Md.RS.O;
            Md.RS.tau = @(M, D) sum(M.Coef ./ Pm.l, 2) .* full(sum(D.IS, 2)) .* sum(M.ES.O.^2, 'all');
            Md.RS.G = @(M, D, ED) -SR(ED * M.ES.O');

            Md.UI = @(X, D) X(D.iu, :, :);
            Md.SI = @(X, D) X(D.is, :, :);
            Md.Predict = @(M, D, phi) (M.SI(M.AS.O, D) + M.UI(M.AU.O, D) + M.B.O + M.UI(M.RU.O, D) * M.EU.O + M.SI(M.RS.O, D) * M.ES.O) * (phi .* M.C + (1 - phi) .* eye(M.Pm.l)); % 
                         
        end
    %% Iteration Parameters
        k = 1;
        t = 1;
        gam = 0;   
        PreLoss = 1e9;

    %% Preloss
        ED = SI(Md.AS.O) + UI(Md.AU.O) + Md.B.O + UI(Md.RU.O) * Md.EU.O + SI(Md.RS.O) * Md.ES.O;
        Loss = - sum(Dt.E .* log(1e-10 + sigm(ED)) +  (1 - Dt.E) .* log(1e-10 + 1 - sigm(ED)), 'all');
        ED = ED * Md.C;
        Loss = Loss - sum(Dt.E .* log(1e-10 + sigm(ED)) +  (1 - Dt.E) .* log(1e-10 + 1 - sigm(ED)), 'all');
        Loss = Loss + 0.5 .* Pm.alpha .* (sum(Md.RU.O.^2, 'all') + sum(Md.RS.O.^2, 'all'));
        Loss = Loss + 0.5 .* Pm.alpha .* (sum(Md.EU.O.^2, 'all') + sum(Md.ES.O.^2, 'all'));
        Loss = Loss + Pm.beta .* (norm(Md.AU.O, 1) + norm(Md.AS.O.^2, 1));
        Loss = Loss ./ (Pm.N * Pm.l);
        PreLoss = Loss;
        fprintf("Iter: 0 Loss: %.6f\t Frac: %.2e\n", PreLoss, 0);
    %% Iteration
        while k < 150;   
            % B
            N = Md.B.O + gam .* (Md.B.O - Md.B.P);
            ED = EG(Md.C, Dt.E, SI(Md.AS.O) + UI(Md.AU.O) + N + UI(Md.RU.O) * Md.EU.O + SI(Md.RS.O) * Md.ES.O);
            Md.B.P = Md.B.O;
            Md.B.O = L1Prox(N, Md.B.tau(Md, Dt), Md.B.G(Md, Dt, ED), Pm.beta);
            % AU 
            N = Md.AU.O + gam .* (Md.AU.O - Md.AU.P);
            ED = EG(Md.C, Dt.E, SI(Md.AS.O) + UI(N) + Md.B.O + UI(Md.RU.O) * Md.EU.O + SI(Md.RS.O) * Md.ES.O);
            Md.AU.P = Md.AU.O;
            Md.AU.O = L1Prox(N, Md.AU.tau(Md, Dt), Md.AU.G(Md, Dt, ED), Pm.beta);

            % AS
            N = Md.AS.O + gam .* (Md.AS.O - Md.AS.P);
            ED = EG(Md.C, Dt.E, SI(N) + UI(Md.AU.O) + Md.B.O + UI(Md.RU.O) * Md.EU.O + SI(Md.RS.O) * Md.ES.O);
            Md.AS.P = Md.AS.O;
            Md.AS.O = L1Prox(N, Md.AS.tau(Md, Dt), Md.AS.G(Md, Dt, ED), Pm.beta);

            % EU
            N = Md.EU.O + gam .* (Md.EU.O - Md.EU.P);
            ED = EG(Md.C, Dt.E, SI(Md.AS.O) + UI(Md.AU.O) + Md.B.O + UI(Md.RU.O) * N + SI(Md.RS.O) * Md.ES.O);
            Md.EU.P = Md.EU.O;
            Md.EU.O = L2Prox(N, Md.EU.tau(Md, Dt), Md.EU.G(Md, Dt, ED), Pm.alpha);
            
            % ES
            N = Md.ES.O + gam .* (Md.ES.O - Md.ES.P);
            ED = EG(Md.C, Dt.E, SI(Md.AS.O) + UI(Md.AU.O) + Md.B.O + UI(Md.RU.O) * Md.EU.O + SI(Md.RS.O) * N);
            Md.ES.P = Md.ES.O;
            Md.ES.O = L2Prox(N, Md.ES.tau(Md, Dt), Md.ES.G(Md, Dt, ED), Pm.alpha);

            % RU
            N = Md.RU.O + gam .* (Md.RU.O - Md.RU.P);
            ED = EG(Md.C, Dt.E, SI(Md.AS.O) + UI(Md.AU.O) + Md.B.O + UI(N) * Md.EU.O + SI(Md.RS.O) * Md.ES.O);
            Md.RU.P = Md.RU.O;
            Md.RU.O = L2Prox(N, Md.RU.tau(Md, Dt), Md.RU.G(Md, Dt, ED), Pm.alpha);
    
            % RS
            N = Md.RS.O + gam .* (Md.RS.O - Md.RS.P);
            ED = EG(Md.C, Dt.E, SI(Md.AS.O) + UI(Md.AU.O) + Md.B.O + UI(Md.RU.O) * Md.EU.O + SI(N) * Md.ES.O);
            Md.RS.P = Md.RS.O;
            Md.RS.O = L2Prox(N, Md.RS.tau(Md, Dt), Md.RS.G(Md, Dt, ED), Pm.alpha);        

            % Loss
            ED = SI(Md.AS.O) + UI(Md.AU.O) + Md.B.O + UI(Md.RU.O) * Md.EU.O + SI(Md.RS.O) * Md.ES.O;
            Loss = - sum(Dt.E .* log(1e-10 + sigm(ED)) +  (1 - Dt.E) .* log(1e-10 + 1 - sigm(ED)), 'all');
            ED = ED * Md.C;
            Loss = Loss - sum(Dt.E .* log(1e-10 + sigm(ED)) +  (1 - Dt.E) .* log(1e-10 + 1 - sigm(ED)), 'all');
            Loss = Loss + 0.5 .* Pm.alpha .* (sum(Md.RU.O.^2, 'all') + sum(Md.RS.O.^2, 'all'));
            Loss = Loss + 0.5 .* Pm.alpha .* (sum(Md.EU.O.^2, 'all') + sum(Md.ES.O.^2, 'all'));
            Loss = Loss + Pm.beta .* (norm(Md.AU.O, 1) + norm(Md.AS.O.^2, 1));
            Loss = Loss ./ (Pm.N * Pm.l);
            % Evaluation
            Frac = abs(PreLoss - Loss) ./ PreLoss;
            if mod(k, 1) == 0 && Pm.verbose; fprintf("Iter: %d Loss: %.6f\t Frac: %.2e\n", k, Loss, Frac); end;
            if Frac < Pm.thres; break; end;
            % Epoch
            k = k + 1;
            t_1 = t;
            t = (1 + sqrt(1 + 4 * t^2)) / 2;
            gam = (t_1 - 1) / t;    
            PreLoss = Loss;
        end
end