% v: for variable
% k: is the damping factor

function inv_J = ve_dampedinv(J, dimension, k1, m0)

    % find the dimension of the operation automatically
    [rows,~] = size(J); 
    
    % find the SVD of J to operate on the singular values
    [U,S,V] = svd(J);
    
    % find the manipulability of the manipulator
    %m = sqrt(det(J'*J))
    m = real(sqrt(det(J*J')))
    
    if (m < m0)
        k2 = k1*(1 - m/m0);
        E = diag(S)./(diag(S).^2+k2);
    else
        E = diag(S)./(diag(S).^2+k1);
    end
    
    % build the modified matrix of singular values
    E = diag(E);
    
    % compute the fixed damped pseudo-inverse
    inv_J = J'*inv(J*J' + E*eye(dimension));  
end