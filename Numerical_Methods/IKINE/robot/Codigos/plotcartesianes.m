%function plotcartesianes(robot)

load Kp1KdVAR250.mat

mida=size(q)
Xn=[];
Yn=[];
Zn=[];
euln=[;;];

for i=1:mida(1)
    T=fkine(robot,q(i,:));
    i/mida(1)
    Xn=[Xn;T(1,4)];
    Yn=[Yn;T(2,4)];    
    Zn=[Zn;T(3,4)];
    euln=[euln tr2eul(T)'];
    
    
end
kk=2;
plot(time,Xn,'r','linewidth',kk)
hold on
plot(time,Yn,'b','linewidth',kk)
plot(time,Zn,'g','linewidth',kk)

load kp1000kd250.mat

mida=size(q)
X=[];
Y=[];
Z=[];

eul=[;;];
for i=1:mida(1)
    T=fkine(robot,q(i,:));
    i/mida(1)
    X=[X;T(1,4)];
    Y=[Y;T(2,4)];    
    Z=[Z;T(3,4)];
    eul=[eul tr2eul(T)'];
    
end
plot(time,X,'m','linewidth',kk)
plot(time,Y,'c','linewidth',kk)
plot(time,Z,'k','linewidth',kk)

title('Posicion XYZ cartesiana con Kp=1000,Kd=Id·250 y Kdnew=diag(3,1,2,1,1,1)·250')
xlabel('Tiempo (s)');
ylabel('posición (m)');
legend('X','Y','Z','Xnew','Ynew','Znew')





figure(2)
plot(time,euln(1,:)*180/pi,'r','linewidth',kk)
hold on
plot(time,euln(2,:)*180/pi,'b','linewidth',kk)
plot(time,euln(3,:)*180/pi,'g','linewidth',kk)
plot(time,eul(1,:)*180/pi,'m','linewidth',kk)
plot(time,eul(2,:)*180/pi,'c','linewidth',kk)
plot(time,eul(3,:)*180/pi,'k','linewidth',kk)
title('Ángulos de Euler ZYZ con Kp=1000,Kd=Id·250 y Kdnew=diag(3,1,2,1,1,1)·250')
xlabel('Tiempo (s)');
ylabel('ángulo (º)');
legend('fi','theta','psi','fi_new','theta_new','psi_new')

