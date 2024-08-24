function [S, dl, dr] = dscale(A)
tol = 1e-15;
% tol1 = 2e-15;

% [U1,S1,V1] = svd(A,'econ');
% s1 = diag(S1);
% tol2 = max(size(A)) * eps(norm(s1,inf));

% tol = min(tol1, tol2);

[m, n] = size(A);
L = zeros(m, n); M = ones(m, n);
S = sign(A); A = abs(A);
idx = find(A > 0.0); L(idx) = log(A(idx));
idx = setdiff(1 : numel(A), idx);
L(idx) = 0; M(idx) = 0;
r = sum(M, 2); c = sum(M, 1);
u = zeros(m, 1); v = zeros(1, n);
dx = 2*tol;
count = 0;
while (dx > tol)
idx = c > 0;
p = sum(L(:, idx), 1) ./ c(idx);
L(:, idx) = L(:, idx) - repmat(p, m, 1) .* M(:, idx);
v(idx) = v(idx) - p; dx = mean(abs(p));
idx = r > 0;
p = sum(L(idx, :), 2) ./ r(idx);
L(idx, :) = L(idx, :) - repmat(p, 1, n) .* M(idx, :);
u(idx) = u(idx) - p; dx = dx + mean(abs(p));
%tol1
%tol2
%tol
%dx
%dx - tol
%    if ((dx - tol) < 5e-16)
%        break
%    end

count = count + 1;
if count >= 1000
    break
end


end

dl = exp(u); dr = exp(v);
S = S.* exp(L);
end
