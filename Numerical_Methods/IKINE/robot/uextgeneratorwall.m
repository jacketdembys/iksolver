function ue=uextgeneratorwall(u,n)

t=u(1);
uc=u(2:n+1);
q=u(n+2:2*n+1);

prova=1;
for i=1:n
   if q(i)<0.9999
       prova=0;
   end       
       
end

v=ones(n,1);
ue=prova*dot(uc,v)*v;
if norm(ue)>0
   hola=1; 
end
%ue=umax*min(u,tmax)/tmax;
%if u<2
%    ue=ones(1,6)*0.0;
%else
%   ue=ones(1,6)*0.1;
%end