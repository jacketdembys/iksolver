function out = observerest2(robot,u)
n=robot.n;

   
uc=u(1:n);
x1=u(n+1:2*n);
x2=u(2*n+1:3*n);
x1e=u(3*n+1:4*n);
x2e=u(4*n+1:5*n);
dx2e=u(5*n+1:6*n);
nest=u(6*n+1:7*n);



K1=100*eye(n);
K2=100*eye(n);

M=inertia(robot,x1);
[U,S,V]=svd(M);
for i=1:n
   %S2(i,i)=1/eigfilter(10,0.001,S(i,i));
   if S(i,i)<0.001
       S2(i,i)=0;
   else
      S2(i,i)=1/S(i,i); 
   end
end

Mi=V*S2*U';
MG=uc-nest;
G=Mi*MG;
%G=y;

dx1e=x2e+K1*(x1-x1e);
dx2e=G+Mi*K2*(x1-x1e);

de=-K2*(x1-x1e);

out=[dx1e;dx2e;de];
