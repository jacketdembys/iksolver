function out = observerest(robot,u,K11,K12,K21,K22)
n=robot.n;

   
uc=u(1:n);
x1=u(n+1:2*n);
x2=u(2*n+1:3*n);
x1e=u(3*n+1:4*n);
x2e=u(4*n+1:5*n);
dx2e=u(5*n+1:6*n);
nest=u(6*n+1:7*n);



%K11=10*eye(n);
%K21=-25*eye(n);
%K12=0*eye(n);
%K22=10*eye(n);

M=inertia(robot,x1);
MG=uc-nest;

%G=sdinverse(M,MG,50);
G=M\MG; 
%[sdinverse(M,MG,20) M\MG]
%G=y;
%Mide=-x2+x2e-dx2e+G;
Ka=1;Kb=1;
de=M*(Ka*(G-dx2e)+Kb*(x2e-x2));


dx1e=x2e+K11*(x1-x1e)+K12*(x2-x2e);
%dx1e=x2+(K11-K12*(K22+eye(n))^(-1)*K21)*(x1-x1e)
%dx1e=x2+K11*(x1-x1e)+K12*(x2-x2e);
dx2e=dx2e+K21*(x1-x1e)+(K22+eye(n))*(x2-x2e);
%dx2e=x2-x2e+dx2e+K21*(x1-x1e)+K22*(x2-x2e);


out=[dx1e;dx2e;de];

