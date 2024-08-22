%function [G,de,Mide] = disturb(u)
function out = inertiacalc(robot,u)
n=robot.n;
q=u(1:n);
y=u(n+1:2*n);


M=inertia(robot,q);
out=M*y;

