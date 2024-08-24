% f is for fixed damping factor
% k: is the damping factor

function inv_J = fie_dampedinv(J, dimension, k1, omega)

    % find the dimension of the operation automatically
    [rows,~] = size(J); 
    
    % find the SVD of J to operate on the singular values
    [U,S,V] = svd(J);
    
    % build the modified matrix of singular values
    E = diag(S)./(diag(S).^2+k1);
    E = diag(E);
    
    % compute the fixed damped pseudo-inverse
    inv_J = J'*inv(J*J' + E*eye(dimension) + omega*eye(dimension));  
    
end