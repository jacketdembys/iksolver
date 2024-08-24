% k: is the damping factor

function inv_J = ivie_dampedinv(J, dimension, k1, e, omega)

    % find the dimension of the operation automatically
    [rows,~] = size(J); 
    
    %{
    % find the SVD of J to operate on the singular values
    [U,S,V] = svd(J);
    
    % find the manipulability of the manipulator
    k2 = k1*e.^2;
    E = diag(S)./(diag(S).^2+k2);
       
    % build the modified matrix of singular values
    E = diag(E);
    
    % compute the fixed damped pseudo-inverse
    inv_J = J'*inv(J*J' + E*eye(dimension) + omega*eye(dimension));  
    %}
    
    %% From author
    n = rows;
    e2 = e;
    
    [UU,SS,VV]=svd(J);
    Ju=zeros(size(J))';

    for i=1:n
        SS(i,i)=SS(i,i)/(SS(i,i)^2+0.5*(e2'*e2)+0.001);
        Ju=Ju+VV(:,i)*UU(:,i)'*SS(i,i);
    end
    
    inv_J = Ju;
    
end