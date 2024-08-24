% k: is the damping factor

function d_theta = select_dampinv2(J, gamma_max, e)

%{
    % find the SVD of J to operate on the singular values
    [U,S,V] = svd(J); 

    % find the rows and cols of J
    [rows,cols] = size(J); 
    
    % number of end-effector 1 for our manipulator case
    num_endEffectors = 1;
    
    % calculate the response vector dTheta that is the SDLS solution
    d_theta = zeros(cols,1);
    
    %diagLength = length(diag(S));
    S_vec = diag(S);    
    diagLength = length(S_vec);  
    
    
    % calculate the norm the vectors in the Jacobian  
    Jnorms = zeros(rows,1);
    for i=1:rows        
        Jnorms(i) = norm(J(i,:),2);
    end
    
    dT = e;
 
     
    for i=1:diagLength
        
        
        %% Based on other implementation
%         dotProdCol = dot(U(:,i),e);
%         alpha = S_vec(i);
        
        
        %% My own implementation section
        % core of the selective damped least-squares
        
        % find sum of N_i
        %N = sum(U(:,i));
        
%         M_each = 0;        
%         for j=1:cols
%             M_each = M_each + V(j,i)*J(i,j);
%         end



        % find sum of the magnitude
        alpha = dot(U(:,i), dT);
        N = abs(alpha);       
        M = sum(abs(V(:,i))*Jnorms(i));
        M = inv(S_vec(i))*M;
        %M = inv(S_vec(i))*(V(:,i)'*J(i,:)');
        gamma_each = min(1, N/M)*gamma_max;  

        % find clamped max phi_i
        % here kept the notation of the paper to relate better
        w = inv(S_vec(i))*alpha*V(:,i);
        %w = e;
        d = gamma_each;
        if (max(abs(w)) <= d)
            phi_each = w;
        else
            phi_each = d*(w/max(abs((w))));
        end
         
        
        % find the inv_J
        d_theta = d_theta + phi_each;
        %d_theta = d_theta + select_damped*V(:,i)*U(:,i)';
    end
    
%     % build the modified matrix of singular values
%     E = diag(S)./(diag(S).^2+k1);
%     E = diag(E);
%     
%     % compute the fixed damped pseudo-inverse
%     inv_J = J'*inv(J*J' + E*eye(dimension));  

%}

    %% From author
    [rows,cols] = size(J); 
    n = rows;
    m = cols;
    e2 = e;
    gammamax=0.5;
    st = 2;
    [UU,SS,VV]=svd(J);
        
        w=zeros(m,1);
        for i=1:m
            
            if (i>n)
                wi=zeros(m,1);
                Ni=0;
                sig=0;
            elseif and(i==2,SS(2,2)==0)
                wi=zeros(m,1);
                Ni=0;
                sig=0;
            else
                sig=1/SS(i,i);
                wi=sig*VV(:,i)*UU(:,i)'*e2;
                Ni=norm(UU(:,i));
            end
            Mi=0;
            for ij=1:m
                Mi=Mi+sig*abs(VV(ij,i))*norm(J(:,i));
            end
            gammai=gammamax*min(1,Ni/Mi);
            
            if max(wi)>gammai
                thi=gammai*wi/max(wi);
            else
                thi=wi;
            end
            w=w+thi;
        end
        if max(w)>gammamax
            dq=gammamax*w/max(w);
        else
            dq=w;
        end
        mu=0;
        %step
        if st==0
            alpha=0.1;
        elseif st==1
            aux=svd(J);
            alpha=0.2*max(min(aux),0.05);
            %alpha=max(min(aux),0.02);
        elseif st==2
            %JJe=J*Ju*e2;
            %alpha=dot(JJe,e2)/dot(JJe,JJe);
            alpha=1;
        end
        
        d_theta = dq;
end