%function [G,de,Mide] = disturb(u)
function out = inertcalc(u)

q=u(1:6);
y=u(7:12);
puma560;


M=inertia(p560,q);
out=M*y;

