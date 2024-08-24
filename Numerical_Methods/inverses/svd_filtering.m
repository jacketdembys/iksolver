% k: is the damping factor

function inv_J = svd_filtering(J)

    % find the dimension of the operation automatically
    [rows,cols] = size(J); 
    
    % find the SVD of J to operate on the singular values
    [U,S,V] = svd(J);
    
    % find the manipulability of the manipulator
    S_diag = diag(S);
    S_diag_0 = 0.01;
    nu = 10;
    inv_J = zeros(cols, rows);
    for i=1:rows
        inv_J = inv_J + ((S_diag(i)^2+nu*S_diag(i) + 2) / (S_diag(i)^3 + nu*S_diag(i)^2 + 2*S_diag(i) + 2*S_diag_0))*V(:,i)*U(:,i)';
    end  
  
end