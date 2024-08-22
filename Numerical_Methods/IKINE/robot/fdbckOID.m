function v=fdbckOID(robot,u)
%u=[q;dq];
n=robot.n;
q=u(1:n);
dq=u(n+1:2*n);



% We compute J, dJ, Ja, dJa
Te=fkine(robot,q);
eul=tr2eul(Te);
Xe=[Te(1:3,4);eul'];
ca=cos(eul(1));
sa=sin(eul(1));
cb=cos(eul(2));
sb=sin(eul(2));
Taux=[0 -sa ca*sb;0 ca ca*sb;1 0 cb];
Ti=[eye(3) zeros(3); zeros(3) pinv(Taux)];
J=jacobn(robot,q);
Ja=Ti*J;
dJ=jacobn_dot(robot,q,dq);
da=Ja(4,:)*dq;
db=Ja(5,:)*dq;

dTiaux=[1/sb*[-sa*cb*da-ca/sb*db -ca*cb*da+sa/sb*db 0];
    [-ca*da -sa*da 0];
    1/sb*[-sa*da-ca*cb/sb*db ca*da-ca/sb*cb*db 0]];
dTi=[eye(3) zeros(3); zeros(3) dTiaux];
dJa=dTi*J+Ti*dJ;

%we compute outputs
dxe=Ja*dq;
dJdq=dJa*dq;
v=[Xe;dxe;dJdq];


%q0 =[0.4456 0.6463 0.7094 0.7547 0.2760 0.6797]
%T0=fkine(p560,q0)
%x0d=[T0(1:3,4);tr2eul(T0)']
%qf =[0.5853    0.2238    0.7513    0.2551    0.5060    0.6991]
%Tf=fkine(p560,qf)
%xf=[Tf(1:3,4);tr2eul(Tf)']
%EULM=[];
%for i=1:26
   %q=outqe(i,1:6);
   %T=fkine(p560,q);
   %EULM=[EULM;tr2eul(T)];  
%end