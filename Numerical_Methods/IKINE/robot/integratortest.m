close all
n=1000

t=0.5;
%function: dexp(x/5)=1/5*exp(1/5*x),
y0=fintegrator(t,1);
y1=fintegrator(t,1);
y2=fintegrator(t,1);
y3=fintegrator(t,1);
dy0=fintegrator(t,0);
dy1=fintegrator(t,0);
dy2=fintegrator(t,0);
dy3=fintegrator(t,0);

DY=dy0;
Y=y0;
h=0.001;
Ytrue=fintegrator(t,1);
T=t;
for i=1:n-1
    
    t=t+h;
    dy0=dy1;
    dy1=dy2;
    dy2=dy3;
    dy3=fintegrator(t,0);
    
    
    y0=y1;
    y1=y2;
    y2=y3;
    y3=integrator(dy0,dy1,dy2,dy3,h,i)+y0;
    
    DY=[DY;dy3];
    Y=[Y;y3];
    Ytrue=[Ytrue;fintegrator(t,1)];
    T=[T;t];
     
end
figure(1)
plot(T,Ytrue,'r','Linewidth',3);
hold on
plot(T,Y,'b','Linewidth',2);

figure(2)
plot(T,Ytrue-Y,'Linewidth',2)