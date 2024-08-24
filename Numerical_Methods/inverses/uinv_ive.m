function Ai = uinv_ive(A, dim, d, e)

    %% scaling decomposition
    [S, dl, dr] = dscale(A);
    diag(dl);
    diag(dr);

    %% error damping on S
    S_inv = ive_dampedinv(S, dim, d, e);

    %% overall inverse
    Ai = S_inv .* (dl * dr)';

end