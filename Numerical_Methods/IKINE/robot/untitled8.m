% augmentant Kd, disminueix l'amplitud d'oscil·lació
n=6

%ALLEST=[]
%ALLEST=[ALLEST estimations]

X1eb=estimations1(:,n+1:2*n);
DX1eb=estimations1(:,2*n+1:3*n);
X2eb=estimations1(:,3*n+1:4*n);
DX2eb=estimations1(:,4*n+1:5*n);
Deb=estimations1(:,1:n);
%y=estimations(:,50:55)

%uc=comprovacio(:,7:12);
%ut=simout(:,4);
%X1eALL=[];
%X1eALL=[X1eALL X1e];

%DX1eALL=[];
%DX1eALL=[DX1eALL DX1e];

%DX2eALL=[];
%DX2eALL=[DX2eALL DX2e];

%X2eALL=[];
%X2eALL=[X2eALL X2e];

%DeALL=[];
%DeALL=[DeALL De];
iaux=5;
%close all
figure(1+iaux)
plot(time,Q,'Linewidth',2)
title('joint position')
hold on
plot(time,X1eb)
legend('q1','q2','q3','q4','q5','q6')

figure(2+iaux)
plot(time,DQ,'Linewidth',2)
title('joint velocity')
hold on
plot(time,X2eb)
legend('q1','q2','q3','q4','q5','q6')

figure(3+iaux)
plot(time,DDQ,'Linewidth',2)
title('joint acceleration')
hold on
plot(time,DX2eb)
legend('q1','q2','q3','q4','q5','q6')

figure(4+iaux)
title('perturbation')
hold on
plot(time,Deb)
legend('q1','q2','q3','q4','q5','q6')


for i=1:size(Deb,1)
    AUX(i,:)=X2eb(i,:)/X2eb(i,1);
    
end
