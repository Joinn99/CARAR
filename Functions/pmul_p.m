function Z = pmul_p(X, Y)
    p = size(Y, 3);
    Z = gpuArray(zeros(size(X, 1), size(Y, 2), p));
    for i=1:length(p)
        Z(:, :, i) = X * Y(:, :, i);
    end
end