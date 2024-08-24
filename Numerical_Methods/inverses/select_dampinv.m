% k: is the damping factor

function dTheta = select_dampinv(J, gamma, e, d)

    % find the SVD of J to operate on the singular values
    [U,S,V] = svd(J); 

    % find the rows and cols of J
    [rows,cols] = size(J); 
    
    % number of end-effector 1 for our manipulator case
    %num_endEffectors = 1;
    
    % calculate the response vector dTheta that is the SDLS solution
    d_theta = zeros(cols,1);
    
    % calculate the norm the vectors in the Jacobian  
    Jnorms = zeros(rows,1);
    for i=1:rows        
        Jnorms(i) = norm(J(i,:),2);
    end
    
    % clamp the updated error values     
    if norm(e, 2) <= d
        dT = e;
    else
        dT = d*(e/norm(e,2));    
    end
    
    % loop over each singular vector
    S_diag = diag(S);
    for i1=1:rows
        wiInv = S_diag(i1);
        if (abs(wiInv) <= 1.0e-10)
            continue
        end
        
        wiInv = 1.0/wiInv;
        alpha = dot(U(:,i1), dT); 
        N = abs(alpha);       
        M = sum(abs(V(:,i1))*Jnorms(i));
        M = M*wiInv;
        
        % scale back the permissible joint angle
        if (N<M)
            gamma = gamma*(N/M);
        end
        
        % calculate dTheta from pure pseudo-inverse considerations
        scale_factor = alpha*wiInv;
        dPretheta = V(:,i)*scale_factor;
        
        % rescale the dTheta values
        maxdPretheta = max(abs(dPretheta)); 
        rescale_factor = (gamma)/(gamma+maxdPretheta);
        dTheta = dPretheta + rescale_factor;
        
        % scale back to not exceed maximum angle change
        maxChange = max(abs(dTheta));
        if (maxChange > gamma)
            dTheta = dTheta * (gamma/(gamma+maxChange));
        end
        
        
    end
    
    
    
    
end