% Compute the Damped Least-Squares Pseudoinverse also called Damped Jacobian.
% Example: inv_J = dampedinv(J, lambda)
% Inputs:  J = input matrix to invert
%          lambda = non-zero (>0) damping factor
% Outputs: inv_J = inverted matrix

function inv_J = dampedinv(J, lambda)
    
    % the damped least-squares 
    [rows,~] = size(J);      
    inv_J = J'*inv(J*J' + lambda^2*eye(rows));  
    
end