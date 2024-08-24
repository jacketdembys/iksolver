% k: is the damping factor

function inv_J = left_pseudoinv(J)

    % find the dimension of the operation automatically
    [rows,cols] = size(J); 
    
    % find the SVD of J to operate on the singular values
    [U,S,V] = svd(J);
    
    % find the manipulability of the manipulator
    S_diag = diag(S);
    inv_J = zeros(cols, rows);
    for i=1:rows
        inv_J = inv_J + (1/(S_diag(i)))*V(:,i)*U(:,i)';
    end  
  
end