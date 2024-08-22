%function [G,de,Mide] = disturb(u)
function out = observer3(u)

x1=u(1:6);
x1e=u(7:12);
x2=u(13:18);
x2e=u(19:24);
dx2e=u(25:30);
tau=u(31:36);
puma560;

K11=20*diag([3 4 2 1 1 1]);
K21=1*eye(6);
K12=1*eye(6);
K22=20*diag([3 4 2 1 1 1]);

M=inertia(p560,x1);
Gaux=(tau-coriolis(p560,x1',x2')'-gravload(p560,x1')'-friction(p560,x2')');
G=M\Gaux;
Mide=(x2-x2e+dx2e-Gaux);
de=M*Mide;

dx1e=x1e+K11*(x1-x1e)+K12*(x2-x2e);
dx2e=G+K21*(x1-x1e)+K22*(x2-x2e)+Mide;


out=[dx1e;dx2e;de];

