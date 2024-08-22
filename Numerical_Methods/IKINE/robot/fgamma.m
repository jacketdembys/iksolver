%function [G,de,Mide] = disturb(u)
function G = fgamma(u)

q=u(1:6);
tau=u(7:12);
dq=u(13:18);
x2e=u(19:24);
dx2e=u(25:30);
puma560;


M=inertia(p560,q);
%Mi=inv(M);
G=M\(tau-coriolis(p560,q',dq')'-gravload(p560,q')'-friction(p560,dq')');
%Mide=(x2-x2e+dx2e-G)
%de=M*Mide;

