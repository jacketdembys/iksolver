
Xe=outxe(:,1:6);
DXe=outxe(:,7:12);
Y=outxe(:,13:18);
Xd=outxd(:,1:6);
DXd=outxd(:,7:12);
DDXd=outxd(:,13:18);
dJdq=outxe(:,19:24)

m=7
My=comp(:,1:m);
n=comp(:,m+1:2*m);
ue=comp(:,2*m+1:3*m);
legend('q1','q2','q3','q4','q5','q6','q7')
