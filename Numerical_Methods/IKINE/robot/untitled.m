Q=estimations(:,1:n);
DQ=estimations(:,n+1:2*n);
X1e=estimations(:,2*n+1:3*n);
DX1e=estimations(:,3*n+1:4*n);
X2e=estimations(:,4*n+1:5*n);
DX2e=estimations(:,5*n+1:6*n);
DDQ=estimations(:,6*n+1:7*n);
De=estimations(:,7*n+1:8*n);
time=estimations(:,8*n+1);

%B=zeros(size(Q));
%A=0;
%figure(1)
%jo=2
%for i=1:size(X1e,1)
%A(i)=(X2e(i,jo)-DQ(i,jo))/(X1e(i,jo)-Q(i,jo));
%plot(time,A,'.')

%B(i,1:7)=X2e(i,1:7)+((K11-K12*(K22+eye(7))^(-1)*K21)*(Q(i,1:7)-X1e(i,1:7))')'-DX1e(i,1:7);

%end
plot(time,B)
lol=['K11' num2str(K11) 'K21' num2str(K11) 'K12' num2str(K12) 'K22' num2str(K22) '.fig']
title 'K11'num2str(K11) 'K21' num2str(K11) 'K12' num2str(K12) 'K22' num2str(K22)]
saveas(h,['K11' num2str(K11(1,1)) 'K21' num2str(K21(1,1)) 'K12' num2str(K12(1,1)) 'K22' num2str(K22(1,1)) '.fig'])


for i=1:3
   filename=['data' num2str(i)];
   dataStruct.(filename)=load([filename '.mat'])
    
end