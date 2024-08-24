function inv_J = mixinv(robot, J)
    
    if(strcmp(robot, 'RRPR'))

        J_s = [zeros(1,5); zeros(6, 1) J];
        W = J_s(1, 1);
        X = J_s(1, 2:5);
        Y = J_s(2:7, 1);
        Z = J_s(2:7, 2:5);

        inv_J_s = comGinv(W, X, Y, Z);

        % remove the zero padding
        J = J_s(2:7, 2:5);
        inv_J = inv_J_s(2:5, 2:7);
    
    elseif(strcmp(robot, 'RRPRR'))
        
        J = [J(:,2:3) J(:,1) J(:,4:5)];

        iW = J(1:3, 1:2);
        iX = J(1:3, 3:5);
        iY = J(4:6, 1:2);
        iZ = J(4:6, 3:5);

        inv_J = comGinv(iW, iX, iY, iZ);

        inv_J = [inv_J(3,:); inv_J(1:2,:); inv_J(4:5,:)];
        
   elseif (strcmp(robot, 'RRPRRRR'))

%         iW = J(1:3, 1:3);
%         iX = J(1:3, 4:7);
%         iY = J(4:6, 1:3);
%         iZ = J(4:6, 4:7);

%         inv_J = comGinv(iW, iX, iY, iZ);    


     
        iW = J(1:3, 1:4);
        iX = J(1:3, 5:7);
        iY = zeros(1,4);
        iZ = zeros(1,3);

        inv_J_t = comGinv(iW, iX, iY, iZ);

        inv_J = inv_J_t(:,1:3);


    end

end