% Retrieve the DH table of a robotic manipulator.
% Example: DH = getDH(robot, Q_initial)
% Inputs:  robot = a string representing the robot to load
%          Q_initial = a vector representing the initial joint
%          configuration of the robot to load
% Outputs: DH = a matrix representing the corresponding DH table

function DH = getDH_rad(robot, Q_initial, unit_chosen)

    
    if (strcmp(robot, 'RRP'))     
        DH = [Q_initial(1),                  0,       1,       0;
              Q_initial(2),                  0,     1.1,    pi/2;
                         0,       Q_initial(3),       0,       0];                     

        % convert the entries of the DH table
        DH(1:2, 2) = DH(1:2, 2) * unit_chosen;
        DH(:, 3) = DH(:, 3) * unit_chosen;
            
    elseif (strcmp(robot, 'RRRRRR'))    
        
        DH = [Q_initial(1),                     0.830,        0.400,       deg2rad(-90);
              Q_initial(2)-deg2rad(90),           0.0,        1.175,       deg2rad(0.0);
              Q_initial(3)+deg2rad(90),           0.0,        0.250,       deg2rad(-90);
              Q_initial(4)-deg2rad(180),        1.125,          0.0,       deg2rad(-90);
              Q_initial(5),                       0.0,          0.0,        deg2rad(90);
              Q_initial(6),                     0.230,          0.0,         deg2rad(0)];

        % convert the entries of the DH table
        DH(:, 2) = DH(:, 2) * unit_chosen;
        DH(:, 3) = DH(:, 3) * unit_chosen; 
        
    elseif (strcmp(robot, 'RRRR'))    
        
        DH = [Q_initial(1),        0.0,      1.0,     deg2rad(0.0);
              Q_initial(2),        0.0,      1.0,     deg2rad(0.0);
              Q_initial(3),        0.0,      1.0,     deg2rad(0.0);
              Q_initial(4),        0.0,      1.0,     deg2rad(0.0);];

        % convert the entries of the DH table
        DH(:, 2) = DH(:, 2) * unit_chosen;
        DH(:, 3) = DH(:, 3) * unit_chosen; 
        
    elseif (strcmp(robot, 'RRRRRRR'))


        DH = [Q_initial(1),    0.333,      0.0,     0;
              Q_initial(2),      0.0,      0.0,  pi/2;
              Q_initial(3),    0.316,      0.0,  pi/2;
              Q_initial(4),      0.0,   0.0825,  pi/2;
              Q_initial(5),    0.384,  -0.0825,  pi/2;
              Q_initial(6),      0.0,      0.0,  pi/2;
              Q_initial(7),    0.107,    0.088,  pi/2];

        %{
        
        DH = [Q_initial(1),             0.0,        0.0,        deg2rad(-90);
              Q_initial(2),             0.0,        0.0,        deg2rad(90);
              Q_initial(3),             0.55,       0.045,      deg2rad(-90);
              Q_initial(4),             0.0,       -0.045,      deg2rad(90);
              Q_initial(5),             0.3,        0.0,        deg2rad(-90);
              Q_initial(6),             0.0,        0.0,        deg2rad(90);
              Q_initial(7),             0.06,       0.0,        deg2rad(0)];

        %}
        
        % convert the entries of the DH table
        DH(:, 2) = DH(:, 2) * unit_chosen;
        DH(:, 3) = DH(:, 3) * unit_chosen; 
        
    elseif (strcmp(robot, 'RRPRRRR'))
        

        DH = [Q_initial(1),             0.0,      0.0,  pi/2;
              Q_initial(2),             0.0,     0.25,  pi/2;
                        0.0,   Q_initial(3),      0.0,   0.0;
              Q_initial(4),             0.0,      0.0,  pi/2;
              Q_initial(5),            0.14,      0.0,  pi/2;
              Q_initial(6),             0.0,      0.0,  pi/2;
              Q_initial(7),             0.0,      0.0,  pi/2];
        %{
        DH = [Q_initial(1),                  0.0,       0.0,        deg2rad(90);
              Q_initial(2),                  0.0,       0.25,        deg2rad(90);
                       0.0,       Q_initial(3),         0.0,         deg2rad(0);
              Q_initial(4),                  0.0,       0.0,        deg2rad(90);
              Q_initial(5),                  0.14,      0.0,        deg2rad(90);
              Q_initial(6),                  0.0,       0.0,        deg2rad(90);
              Q_initial(7),                  0.0,       0.0,         deg2rad(0)];
        %}
          
         
        % convert the entries of the DH table
        DH(1:2, 2) = DH(1:2, 2) * unit_chosen;
        DH(4:7, 2) = DH(4:7, 2) * unit_chosen;
        DH(:, 3) = DH(:, 3) * unit_chosen;   
        
        
        
    elseif (strcmp(robot, 'RPRRRRRRR'))
        
        DH = [Q_initial(1),                  0.0,               0.0,          deg2rad(0);
                       0.0,                  0.38,      Q_initial(2)+0.15,    deg2rad(0);
                       %0.0,                  0.38,             0.15,          deg2rad(0);
              Q_initial(3),                  0.333,             0.0,          deg2rad(0);
              Q_initial(4),                  0.0,               0.0,        deg2rad(-90);
              Q_initial(5),                  0.316,             0.0,         deg2rad(90);
              Q_initial(6),                  0.0,            0.0825,         deg2rad(90);
              Q_initial(7),                  0.384,         -0.0825,        deg2rad(-90);
              Q_initial(8),                  0.0,               0.0,         deg2rad(90);
              Q_initial(9),                  0.107,           0.088,         deg2rad(90);];
          
         
        % convert the entries of the DH table
        DH(1, 2) = DH(1, 2) * unit_chosen;
        DH(3:7, 2) = DH(3:7, 2) * unit_chosen;
        DH(:, 3) = DH(:, 3) * unit_chosen;   
        
    elseif (strcmp(robot, 'RRVPRVRVPRR'))
        
        DH = [Q_initial(1),                   0.77,                  0.0,        pi/2;
              Q_initial(2),                   0.0,                   0.557,      pi/2;
                       0.0,                   2.670,                 0.0,         0.0;         % This is a virtual joint   
                       0.0,           Q_initial(4),                  0.0,       -pi/2;
              Q_initial(5),                   0.0,                   0.485,      pi/2;
                       0.0,                   1.8,                   0.0,        pi/2;        % This is a virtual joint 
              Q_initial(7),                   1.425,                 0.0,       -pi/2;
                       0.0,                   0.016359,              0.0,        0.0;         % This is a virtual joint 
                       0.0,          Q_initial(9),                   0.0,        0.0;
              Q_initial(10),                   0.663,                 0.0,        pi/2;
              Q_initial(11),                   1.138,                 0.464,      pi/2];
          
         
        % convert the entries of the DH table
        DH(1:3, 2) = DH(1:3, 2) * unit_chosen;
        DH(5:8, 2) = DH(5:8, 2) * unit_chosen;
        DH(10:11, 2) = DH(10:11, 2) * unit_chosen;
        DH(:, 3) = DH(:, 3) * unit_chosen;      
        
    end     

end