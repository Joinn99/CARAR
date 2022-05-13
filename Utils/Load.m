function [Train, Eval] = Load(Dataset, TI, GPU)
    %% Data loading function

    load("Data/" + Dataset + ".mat");
    m = max(is);
    n = max(iu);
    d = size(D, 2);
    l = size(E, 2);

    Itrain = find(mod(1:size(D,1), 5) ~= TI);
    Itest = find(mod(1:size(D,1), 5) == TI);

    Train.is = is(Itrain)';
    Eval.is = is(Itest)';
    Train.IS = sparse(double(Train.is), double(1:length(Itrain)), double(1), double(m), length(Itrain));

    Train.iu = iu(Itrain)';
    Eval.iu = iu(Itest)';
    Train.IU = sparse(double(Train.iu), double(1:length(Itrain)), double(1), double(n), length(Itrain));

    Train.D = double(D(Itrain, :));
    Train.E = sign(E(Itrain, :));
    Eval.D = double(D(Itest, :));
    Eval.E = sign(E(Itest, :));

    if gpuDeviceCount > 0 && GPU
        Train.IS = gpuArray(Train.IS);
        Train.IU = gpuArray(Train.IU);
        Train.D = gpuArray(Train.D);
        Train.E = gpuArray(Train.E);
    end
end