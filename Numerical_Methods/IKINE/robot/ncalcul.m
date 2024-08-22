function out = ncalcul(robot,u)
n=robot.n;
q=u(1:n);
dq=u(n+1:2*n);


out=coriolis(robot,q',dq')'+gravload(robot,q')';%+friction(robot,dq')';

