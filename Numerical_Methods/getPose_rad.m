% Retrieve the end-effector pose form the total transformation matrix T
% of a robotic manipulator.
% Example: D = getPose(T, n)
% Inputs:  T = transformation matrix
%          n = number of rows to get in the pose vector
% Outputs: D = Pose vector

function D = getPose_rad(T, n)

    %T = round(T, 6);

    D_current = zeros(n, 1); 
    % Retrieve the end-effector position
    D_current(1) = T(1,4);
    D_current(2) = T(2,4);
    D_current(3) = T(3,4);  
    
    % Retrieve the end-effector orientation  --- Roll Pitch Yaw
    %% pose_set_1
    D_current(4) = atan(T(2,1)/T(1,1));
    D_current(5) = atan(-T(3,1)/(T(1,1)*cos(D_current(4))+T(2,1)*sin(D_current(4))));
    D_current(6) = atan((-T(2,3)*cos(D_current(4))+T(1,3)*sin(D_current(4)))/(T(2,2)*cos(D_current(4))-T(1,2)*sin(D_current(4))));    
    
    
    %% pose_set_2
    %{
    D_current(4) = atan2(T(2,1), T(1,1));
    D_current(5) = atan2(-T(3,1), sqrt(T(3,2)^2 + T(3,3)^2));
    D_current(6) = atan2(T(3,2), T(3,3));
    %}
    
    %% pose_set_3
    %{
    D_current(4) = atan2(T(3,2), T(3,3));
    D_current(5) = atan2(-T(3,1), sqrt(T(3,2)^2 + T(3,3)^2));
    D_current(6) = atan2(T(2,1), T(1,1));
    %}
    
    
    
    %% avoid numberical precision errorsD = D_current;
    D = D_current(1:n,1);
    
end