function Md = CARAR_C(Dt, Md, Pm)
    % CARAR: Self-Representation Correlation Mapping
    % Minimize 1/2*tr(C'*B*C) + tr(A'*C)
    % Subject to diag(C) = 0, C'1 = 1, C >= 0
    % Where: B = E' * E | A = -B + Lambda1 * 1
    % Update parameters: C
    % E: R(N*l)                 Review aspect labels
    % N: Sample number          l: Label number
    % ------------------------------------------------------------
    
    %% Initialization
    %% Global Parameters
    k = 1;
    t = ones(1, 1, Pm.l, Pm.type);                  % Nestrov Acceleration
    gamma = zeros(1, 1, Pm.l, Pm.type);             % Momentum
    cr = 1e9;                                       % Combined Residual
    eta = 1 - 1e-5;                                 % Restart Condition
    tol = 1e-7;                                     % Converge Criteria
    Converge = false(1, 1, Pm.l, Pm.type);          % Converge Indicator
    %% Fixed Parameters
    Lambda1 = 1;
    Crho = 1;
    CReg = 'l1';
    %% Output
    C = reshape(1/(Pm.l-1) .* (ones(Pm.l, Pm.type)-eye(Pm.l, Pm.type)), [Pm.l, 1, Pm.l]);
    Z = C;
    Z_1 = Z;
    U = zeros(size(C), Pm.type);
    U_1 = U;
    %% Pre-calculation
    % diag(C) = 0 & e'C = e' --> G_ic_i = h_i
    G = reshape([ones(Pm.l, Pm.type)-eye(Pm.l, Pm.type), eye(Pm.l, Pm.type)]', [Pm.l, 2, Pm.l]);
    h = zeros(2, 1, Pm.type) + [1; 0];
    B = (Dt.E' * Dt.E) ./ (Pm.N - 1);
    B = B + B';
    if CReg == "fro"
        A = reshape(-B, [Pm.l, 1, Pm.l]);
        B = B + 2 .* Lambda1 .* eye(Pm.l, Pm.type);
    else
        A = reshape(Lambda1 - B, [Pm.l, 1, Pm.l]);
    end
    BI = (B + Crho .* eye(Pm.l, Pm.type))^-1;
    BI2 = pmul(BI, pmul(G, pagefun(@inv, pmul(ptrans(G), pmul(BI, G)))));
    BI1 = BI - pmul(BI2, pmul(ptrans(G), BI));
    clear G BI;
    %% Iteration
    while k < 100000
        %% Update
        Z_ = Z + gamma .* (Z - Z_1);
        Z_1 = Z;
        U_ = U + gamma .* (U - U_1);
        U_1 = U;
        C = pmul(BI2, h) - pmul(BI1, A - Crho .* (Z_ - U_));
        Z = max(C + U_, 0);
        U = U_ + C - Z;
        %% Residual
        pr = Crho .* FN2(C - Z);       % Primal residual
        dr = FN2(Crho .* (Z - Z_1));   % Dual residual
        if mod(k, 20) == 0 && Pm.verbose
            fprintf("Iter %d: P/D Residual:%.4e/%.4e Conv: %d\n", k, gather(mean(pr, 'all')), gather(mean(dr,'all')), gather(sum(Converge, 'all')));
        end
        %% Convergence Checking
        Converge = (pr < (tol ./ Pm.l)) & (dr < (tol ./ Pm.l));
        if all(Converge)
            break;
        end
        %% Update Acceleration Parameters with Restart Strategy
        k = k + 1;
        Restart = (pr + dr) > (eta .* cr);
        t_1 = Restart .* 1 + (~Restart) .* t;
        t = Restart .* 1 + (~Restart) .* ((1 + sqrt(1 + 4 .* t.^2)) ./ 2);
        gamma = (t_1 - 1) ./ t;
        cr = Restart .* (cr ./ eta) + (~Restart) .* (pr + dr);
    end
    %% End
    Md.C = squeeze(C);
end