function Ai = uinv(A)
[S, dl, dr] = dscale(A);
diag(dl);
diag(dr);
Ai = pinv(S) .* (dl * dr)';
end