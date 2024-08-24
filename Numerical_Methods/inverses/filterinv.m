function inv_J = filterinv(J, lambda, dimension)
    
    [rows,~] = size(J);     
    [U,S,V] = svd(J);
    u = U(:,dimension);
    inv_J = J'*inv(J*J' + lambda^2*u*u');    
    
end