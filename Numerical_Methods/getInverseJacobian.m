function inv_J = getInverseJacobian(J, inverse_chosen, e, robot)
    
    dim = length(6);

    %% use the Moore-Penrose (MP) inverse
    if strcmp(inverse_chosen, "MP")                   
        inv_J = pinv(J);
        
    %% use the Unit-Consistent (UC) inverse
    elseif strcmp(inverse_chosen, "UC")              
        inv_J = uinv(J);
        
    %% use the Jacobian Transpose (JT) inverse
    elseif (inverse_chosen == "JT")                  
        inv_J = J';
        
    %% use the MX inverse 
    elseif strcmp(inverse_chosen, 'MX')              
        %inv_J = mixinv(robot, J); 
        
        if strcmp(robot, 'RRPRRRR')
            iW = J(1:3, 1:3);
            iX = J(1:3, 4:7);
            iY = J(4:6, 1:3);
            iZ = J(4:6, 4:7);       
            inv_J = comGinv(iW, iX, iY, iZ);
        elseif strcmp(robot, 'RRRRRRR')        
            iW = J(1:3, 1:4);
            iX = J(1:3, 5:7);
            iY = J(4:6, 1:4);
            iZ = J(4:6, 5:7);       
            inv_J = comGinv(iW, iX, iY, iZ);
        elseif strcmp(robot, 'RRVPRVRVPRR')  
            
            iW = J(1:3, 1:9);
            iX = J(1:3, 10:11);
            iY = J(4:6, 1:9);
            iZ = J(4:6, 10:11);
            inv_J = comGinv(iW, iX, iY, iZ);
            
            %{            
            J(:,[3,6,8]) = [];
            iW = J(1:3, 1:6);
            iX = J(1:3, 7:8);
            iY = J(4:6, 1:6);
            iZ = J(4:6, 7:8);
            inv_J = comGinv(iW, iX, iY, iZ);
            
            inv_J = [inv_J(1,:);
                     inv_J(2,:);
                     zeros(1,6);
                     inv_J(3,:);
                     inv_J(4,:);
                     zeros(1,6);
                     inv_J(5,:);
                     zeros(1,6);
                     inv_J(6,:);
                     inv_J(7,:);
                     inv_J(8,:)];
            %}
        end   
        
    %% use the Damped Jacobian
    elseif strcmp(inverse_chosen, 'JD')               
        inv_J = dampedinv(J, 0.005);
        
    %% use the Filtered Jacobian
    elseif strcmp(inverse_chosen, 'JF')                          
        inv_J = filterinv(J, 4*0.005, dim); 
        
    %% use the error damping 
    %{
    elseif strcmp(inverse_chosen, 'FED')
        inv_J = fe_dampedinv(J, dim, 1);
    elseif strcmp(inverse_chosen, 'VED')
        inv_J = ve_dampedinv(J, dim, 1, 1); 
    %}
    elseif strcmp(inverse_chosen, 'ED')
        inv_J = ive_dampedinv(J, dim, 0.5, e);
    %{
    elseif strcmp(inverse_chosen, 'FIED')
        inv_J = fie_dampedinv(J, dim, 1, 0.01);
    elseif strcmp(inverse_chosen, 'VIED')
        inv_J = vie_dampedinv(J, dim, 1, 1, 0.01); 
    %}
    elseif strcmp(inverse_chosen, 'IED')
        inv_J = ivie_dampedinv(J, dim, 1, e, 0.01);
    elseif strcmp(inverse_chosen, 'SVF')
        inv_J = svd_filtering(J);
    elseif strcmp(inverse_chosen, 'LP')
        inv_J = left_pseudoinv(J);
    elseif strcmp(inverse_chosen,  'UCED')
        inv_J = uinv_ive(J, dim, 0.5, e);
    end

end