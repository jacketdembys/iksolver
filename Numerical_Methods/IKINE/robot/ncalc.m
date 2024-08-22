%function [G,de,Mide] = disturb(u)
function out = ncalc(u)

q=u(1:6);
dq=u(7:12);

ddq=u(13:18);
if dd1(6)>1000
    hola=1;
end

puma560;


out=coriolis(p560,q',dq')'+gravload(p560,q')'+friction(p560,dq')';

