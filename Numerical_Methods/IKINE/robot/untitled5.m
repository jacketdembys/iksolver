nf=size(Q,1)
df=De(nf,:);
M=inertia(p560,Q(nf,:));
x2=DQ(nf,:);
x1=Q(nf,:);
x2e=X2e(nf,:);
x1e=X1e(nf,:);
dx2e=DX2e(nf,:);
x3=DDQ(nf,:)

tau=aux(size(aux,1),1:6);
nmatrix=aux(size(aux,1),7:12);

RES=[df' M*x2' M*x2e' M*dx2e' (tau'-nmatrix') tau' nmatrix']

Q=estimations(:,1:6);
DQ=estimations(:,7:12);
X1e=estimations(:,13:18);
DX1e=estimations(:,19:24);
X2e=estimations(:,25:30);
DX2e=estimations(:,31:36);
DDQ=estimations(:,37:42);
De=estimations(:,43:48);
time=estimations(:,49);


Q=simout(:,1:7);
dQ=simout(:,8:14);
ddQ=simout(:,15:21);
uc=simout(:,22:28);
t=simout(:,29);
Qd=simout(:,30:36);
dQd=simout(:,37:43);


n=aux(:,8:14);
PID=aux(:,1:7);
