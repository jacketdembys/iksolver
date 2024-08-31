%function [G,de,Mide] = disturb(u)
function out = observer2(robot,u)
%tau=u(1:6);
n=size(u,1);

if n<36
    u=[u;zeros(36-n,1)];
end
    

x1=u(1:6);
x2=u(12:17);
x1e=u(13:18);
x2e=u(19:24);
dx2e=u(25:30);
y=u(31:36);

K11=1000*diag([1 1 1 1 1 1]);
K21=100*eye(6);
K12=0*eye(6);
K22=100*diag([1 1 1 1 1 1]);


M=inertia(robot,x1);
%G=dx2e-M\dist;
%Gaux=(tau-coriolis(robot,x1',x2')'-gravload(robot,x1')'-friction(robot,x2')');
G=y;
Mide=x2-x2e+dx2e-G;
de=M*Mide;

dx1e=x2e+K11*(x1-x1e)+K12*(x2-x2e);
dx2e=G+K21*(x1-x1e)+K22*(x2-x2e)+Mide;


out=[dx1e;dx2e;de];
