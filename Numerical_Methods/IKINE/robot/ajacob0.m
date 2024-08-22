
function J0 = ajacob0(robot, q,opt)

	% opt=0 --> RPY
    % opt =1 --> EUL
	Jn = jacobn(robot, q);	% Jacobian from joint to wrist space

	%
	%  convert to Jacobian in base coordinates
	%
	Tn = fkine(robot, q);	% end-effector transformation
	R = t2r(Tn);
	J0 = [R zeros(3,3); zeros(3,3) R] * Jn;
    
    
    if opt==0
        rpy = tr2rpy( fkine(robot, q) );
        B = rpy2jac(rpy);
        if rcond(B) < eps,
            error('Representational singularity');
        end
        J0 = blkdiag( eye(3,3), inv(B) ) * J0;
    elseif opt==1
        eul = tr2eul( fkine(robot, q) );
        B = eul2jac(eul);
        if rcond(B) < eps,
            error('Representational singularity');
        end
        J0 = blkdiag( eye(3,3), inv(B) ) * J0;
    end

