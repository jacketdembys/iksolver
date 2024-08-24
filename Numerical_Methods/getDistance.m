function performed_distance = getDistance(D_current, D_current_p, robot, performed_distance)    

    if strcmp(robot, "RRP")
        performed_distance = performed_distance + sqrt(sum((D_current - D_current_p).^2));
    else
        performed_distance = performed_distance + sqrt(sum((D_current(1:3) - D_current_p(1:3)).^2));
    end

end