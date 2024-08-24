function displayJoints(Q_current, robot)
    
    fprintf("\nCurrent joint configuration:")
    for c=1:length(robot)
        if (strcmp(robot(c), 'R'))
            fprintf("\ntheta_%d = %f", c, Q_current(c));
        elseif (strcmp(robot(c), 'P'))            
            fprintf("\nd_%d = %f", c, Q_current(c));
        end
    end
    fprintf("\n")

end