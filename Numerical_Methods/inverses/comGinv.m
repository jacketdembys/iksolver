function Ai = comGinv(W, X, Y, Z)
invW = uinv(W);
invZ = pinv(Z);
Ai = [uinv(W - X*invZ*Y), -invW*X*pinv(Z-Y*invW*X);
    -invZ*Y*uinv(W - X*invZ*Y), pinv(Z - Y*invW*X)];
end