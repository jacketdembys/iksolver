function J1=rcpinv(Q,W)
%calculation of Q^(+W)
[U,S,V]=svd(W);
Qu=U'*Q;
J1=cpinv(Qu,S)*U';
%REVISAR!!!
