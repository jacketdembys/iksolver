function ue=uextgeneratorwam(u)
%umax=ones(1,6)*0.1;
%tmax=2;
%ue=umax*min(u,tmax)/tmax;
if u<2
    ue=ones(1,7)*0.0;
else
   ue=ones(1,7)*0.1;
end
hola=1;