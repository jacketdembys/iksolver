%function [G,de,Mide] = disturb(u)
function out = observer(robot,u)
n=size(u,1)/6;

   
tau=u(1:n);
x1=u(n+1:2*n);
x2=u(2*n+1:3*n);
x1e=u(3*n+1:4*n);
x2e=u(4*n+1:5*n);
dx2e=u(5*n+1:6*n);



K11=10*eye(n);
K21=0*eye(n);
K12=0*eye(n);
K22=20*eye(n);

M=inertia(robot,x1);

MG=(tau-coriolis(robot,x1',x2')'-gravload(robot,x1')'-friction(robot,x2')');


G=M\MG;







%G=y;
Mide=x2-x2e+dx2e-G;
de=M*Mide;

dx1e=x2e+K11*(x1-x1e)+K12*(x2-x2e);
dx2e=G+K21*(x1-x1e)+K22*(x2-x2e)+Mide;


out=[dx1e;dx2e;de];

