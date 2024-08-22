function ue=uextgenerator(u,n)
%umax=[2 2 2 1 1 0.5 0.1]*0.005;
umax=[2 2 2 1 0.5 0.5 0.002]*0.1;
tmax=1;
%umax=[2 2 1 1 0.5 0.1]*0.05;
%ue=umax;
if and(u<0.3,u>0.15)
    ue=umax-(u-0.15)/0.35*umax;
elseif u<0.15
    ue=umax;
elseif and(u>0.3,u<0.6)
    ue=umax*(u-0.3)/0.3;
else
    ue=umax;
end
ue=ue*0;
   %ue=ue; %ue=2*umax/tmax*0;
   %ue=umax;
   