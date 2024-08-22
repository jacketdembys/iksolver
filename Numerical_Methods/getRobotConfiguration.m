function r = getRobotConfiguration(robot_chosen, unit_chosen, DH)

    if strcmp(robot_chosen, "RRRRRRR")
        
        % links: alpha, a, theta, d
        
        %{
        L1 = link([-pi/2 0       0   0       0],'standard');
        L2 = link([ pi/2 0       0   0       0],'standard');
        L3 = link([-pi/2 0.045   0   0.55    0],'standard');
        L4 = link([ pi/2 -0.045  0   0       0],'standard');
        L5 = link([-pi/2 0       0   0.3     0],'standard');
        L6 = link([ pi/2 0       0   0       0],'standard');
        L7 = link([    0 0       0   0.06    0],'standard'); 
        %}
        
        
        
        
        L1 = link([DH(1, 4)    DH(1, 3)     0   DH(1, 2)    0],'standard');
        L2 = link([DH(2, 4)    DH(2, 3)     0   DH(2, 2)    0],'standard');
        L3 = link([DH(3, 4)    DH(3, 3)     0   DH(3, 2)    0],'standard');
        L4 = link([DH(4, 4)    DH(4, 3)     0   DH(4, 2)    0],'standard');
        L5 = link([DH(5, 4)    DH(5, 3)     0   DH(5, 2)    0],'standard');
        L6 = link([DH(6, 4)    DH(6, 3)     0   DH(6, 2)    0],'standard');
        L7 = link([DH(7, 4)    DH(7, 3)     0   DH(7, 2)    0],'standard'); 
        
        
        r = robot({L1 L2 L3 L4 L5 L6 L7}); 
        
        
    elseif strcmp(robot_chosen, "RRPRRRR")
        
        % links: alpha, a, theta, d       
        L1 = link([DH(1, 4)    DH(1, 3)     0   DH(1, 2)    0],'standard');
        L2 = link([DH(2, 4)    DH(2, 3)     0   DH(2, 2)    0],'standard');
        L3 = link([DH(3, 4)    DH(3, 3)     0          0    1],'standard');
        L4 = link([DH(4, 4)    DH(4, 3)     0   DH(4, 2)    0],'standard');
        L5 = link([DH(5, 4)    DH(5, 3)     0   DH(5, 2)    0],'standard');
        L6 = link([DH(6, 4)    DH(6, 3)     0   DH(6, 2)    0],'standard');
        L7 = link([DH(7, 4)    DH(7, 3)     0   DH(7, 2)    0],'standard'); 
        
        
        r = robot({L1 L2 L3 L4 L5 L6 L7}); 
        
     elseif strcmp(robot_chosen, "RPRRRRRRR")
        
        % links: alpha, a, theta, d       
        L1 = link([DH(1, 4)    DH(1, 3)     0   DH(1, 2)    0],'standard');
        L2 = link([DH(2, 4)    DH(2, 3)     0   DH(2, 2)    1],'standard');
        L3 = link([DH(3, 4)    DH(3, 3)     0   DH(3, 2)    0],'standard');
        L4 = link([DH(4, 4)    DH(4, 3)     0   DH(4, 2)    0],'standard');
        L5 = link([DH(5, 4)    DH(5, 3)     0   DH(5, 2)    0],'standard');
        L6 = link([DH(6, 4)    DH(6, 3)     0   DH(6, 2)    0],'standard');
        L7 = link([DH(7, 4)    DH(7, 3)     0   DH(7, 2)    0],'standard');
        L8 = link([DH(8, 4)    DH(8, 3)     0   DH(8, 2)    0],'standard');
        L9 = link([DH(9, 4)    DH(9, 3)     0   DH(9, 2)    0],'standard'); 
        
        
        r = robot({L1 L2 L3 L4 L5 L6 L7 L8 L9}); 
        
     elseif strcmp(robot_chosen, "RRP")
        
        % links: alpha, a, theta, d       
        L1 = link([DH(1, 4)    DH(1, 3)     0   DH(1, 2)    0],'standard');
        L2 = link([DH(2, 4)    DH(2, 3)     0   DH(2, 2)    0],'standard');
        L3 = link([DH(3, 4)    DH(3, 3)     0          0    1],'standard');
        
        
        r = robot({L1 L2 L3}); 

    elseif strcmp(robot_chosen, "RRVPRVRVPRR")
        
        % links: alpha, a, theta, d       
        L1 = link([DH(1, 4)         DH(1, 3)        0   DH(1, 2)    0   0],'standard');
        L2 = link([DH(2, 4)         DH(2, 3)        0   DH(2, 2)    0   pi/2],'standard');
        L3 = link([DH(3, 4)         DH(3, 3)        0   DH(3, 2)    0   0],'standard'); % this is a virtual link
        L4 = link([DH(4, 4)         DH(4, 3)        0          0    1   0],'standard');
        L5 = link([DH(5, 4)         DH(5, 3)        0   DH(5, 2)    0   0],'standard');
        L6 = link([DH(6, 4)         DH(6, 3)        0   DH(6, 2)    0   pi/2],'standard'); % this is a virtual link
        L7 = link([DH(7, 4)         DH(7, 3)        0   DH(7, 2)    0   0],'standard'); 
        L8 = link([DH(8, 4)         DH(8, 3)        0   DH(8, 2)    0   0],'standard'); % this is a virtual link
        L9 = link([DH(9, 4)         DH(9, 3)        0          0    1   0],'standard');
        L10 = link([DH(10, 4)       DH(10, 3)       0   DH(10, 2)    0  0],'standard');
        L11 = link([DH(11, 4)       DH(11, 3)       0   DH(11, 2)    0  pi/2],'standard');
        
        
        r = robot({L1 L2 L3 L4 L5 L6 L7 L8 L9 L10 L11});
        


    end

end