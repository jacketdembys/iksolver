function out=jacobderivblock(robot,u)
n=robot.n;
q=u(1:n);
dq=u(n+1:2*n);

out=jacobn_dot(robot,q,dq)*dq;