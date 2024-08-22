tic
t=0;
n=6;
%close all
%WAM;
%robot=wam;
robot=p560;
tfinal=1;
niters=2000;
h=tfinal/niters;


dy0=zeros(2*n,1);
dy1=zeros(2*n,1);
dy2=zeros(2*n,1);
dy3=zeros(2*n,1);
ddq0=zeros(n,1);
ddq1=zeros(n,1);
ddq2=zeros(n,1);
ddq3=zeros(n,1);
dq0=zeros(n,1);
dq1=zeros(n,1);
dq2=zeros(n,1);
dq3=zeros(n,1);

xe0=[q0;dq0];
xe1=zeros(2*n,1);
xe2=zeros(2*n,1);
xe3=zeros(2*n,1);

%PROXYS
%p0=dqd(1,:)';
%p1=dqd(1,:)';
%p2=dqd(1,:)';
%p3=dqd(1,:)';
%dp0=ddqd(1,:)';
%dp1=ddqd(1,:)';
%dp2=ddqd(1,:)';
%dp3=ddqd(1,:)';




q0=ones(n,1)*0.2;
qa0=q0;
qa1=q0;
qa2=q0;
qa3=q0;
qf=ones(n,1)*0.4;
%q0=zeros(n,1);
x1e=q0;
dx1e=zeros(n,1);
dq0=zeros(n,1);
x2e=dq0;
dx2e=zeros(n,1);
q=q0;
dq=dq0;
uest=zeros(n,1);
[qd,dqd,ddqd] = jtraj(q0, qf, niters);
    dx1eold=dx1e;
    dx2eold=dx2e;
    uestold=uest;


    SX1e=[];
    SX2e=[];

    SdX1e=[];
    SdX2e=[];  
    
    SQ=[];
    SQd=[];
    
    SdQ=[];
    SdQd=[];  
    
    SddQ=[];
    SddQd=[];

    Suext=[];
    Sfext=[];
    time=[];
    e1=zeros(n,1);
    y=zeros(n,1);
   
        
                %kp = diag([ 900  2500 600  500   50   50    8])/100;
                %kp=diag([ 900  1000 600  500   50   50])/100
                ki = diag([2.5    5   2 0.5  0.5  0.5  0.1]);

                %kd =kp.^0.5/2;% diag([10   20    5    2  0.5  0.5 0.05]);
kd=Kd;
kp=Kp;
                control_signal_limit =[25 20 15 15 5 5 5]*2;
                interror=0;
for i=1:niters
    t0=t;
    t=i*tfinal/niters;
    fext=0*uextgenerator(t,7)';
    n=robot.n;
    M=inertia(robot, q);
   % Kp=M;
    
    
    N= coriolis(robot, q', dq')'+gravload(robot,q')';

    %e=(qd(i,:)'-q);
        %interror=interror+ki*h*e0;
    %y=kp*e+interror+kd*(e-e0)/h;
    
    
    y=ddqd(i,:)'+Kd*(dqd(i,:)'-dq)+Kp*(qd(i,:)'-q);

    for iaux=1:n
        if abs(y(iaux))>control_signal_limit(iaux)
            y(iaux)=y(iaux)/abs(y(iaux))*control_signal_limit(iaux);
        end
    end
    e0=e;
    %tic
    %sim('robotsim.mdl')
   
    uc=M*y+N+uest;
    
    out=observerest(robot,[uc;q;dq;x1e;x2e;dx2e;N],K11,K12,K21,K22);
    fcut=2;
    invtau=2*pi*fcut;
    a=h/(invtau+h);
    dx1e=(1-a)*out(1:n)+a*dx1eold;
    dx2e=(1-a)*out(n+1:2*n)+a*dx2eold;
    uest=0*((1-a)*out(2*n+1:3*n)+a*uestold);
    dx1eold=dx1e;
    dx2eold=dx2e;
    uestold=uest;
    xe0=xe1;
    xe1=xe2;
    xe2=xe3;
    dy0=dy1;
    dy1=dy2;
    dy2=dy3;
    dy3=[dx1e;dx2e];
    xe3=integrator(dy0,dy1,dy2,dy3,h,10)+xe0;
    x1e=xe3(1:n);
    x2e=xe3(n+1:2*n);
    
    
    
    ddq0=ddq1;
    ddq1=ddq2;
    ddq2=ddq3;
    ddq3=M\(uc-fext-N);
    dq0=dq1;
    dq1=dq2;
    dq2=dq3;
    dq3=integrator(ddq0,ddq1,ddq2,ddq3,h,10)+dq0;
    qa0=qa1;
    qa1=qa2;
    qa2=qa3;
    qa3=integrator(dq0,dq1,dq2,dq3,h,10)+qa0;
    q=qa3;
    dq=dq3;
    ddq=ddq3;
   


    

    SX1e=[SX1e;x1e'];
    SX2e=[SX2e;x2e'];
    SdX1e=[SdX1e;dx1e'];
    SdX2e=[SdX2e;dx2e'];    
    SQ=[SQ;q'];
    SQd=[SQd;qd(i,:)];
    SdQ=[SdQ;dq'];
    SdQd=[SdQd;dqd(i,:)];    
    SddQ=[SddQ;ddq'];
    SddQd=[SddQd;ddqd(i,:)];   
    Suext=[Suext;uest'];
    Sfext=[Sfext;fext'];
    time=[time;t];
    
    holaplot=1;
    iplots=6;
    if holaplot==1
        
        figure(iplots)
        hold on
        plot(t,uest(1),'b')
        plot(t,fext(1),'r')
        
        figure(iplots+1)
        hold on
        plot(t,q(1),'b')
        plot(t,qd(i,1),'r')
        plot(t,x1e(1,1),'k')
        
        figure(iplots+2)
        hold on
        plot(t,dq(1),'b')
        plot(t,dqd(i,1),'r')
        plot(t,dx1e(1,1),'g')
        plot(t,x2e(1,1),'k')
        
        figure(iplots+3)
        hold on
        plot(t,y(1),'g')
        plot(t,ddq(1),'b')
        plot(t,ddqd(i,1),'r')
        plot(t,dx2e(1,1),'k')
        
    end
    
end

T=1;
%spectral;


%figure(iaux+2)
%plot(time,Suext)
%hold on
%plot(time,Sfext,'Linewidth',2)
%title 'external force'

%figure(iaux+3)
%plot(time,SQ)
%hold on
%plot(time,SQd,'Linewidth',2)
%title 'joint position'
%close all;+
plotaltres=0;
if plotaltres==1
figure(iaux)
                hplot=plot(time,SdX2e(:,1),'b')
                hold on
                plot(time,SdX2e(:,2),'r')
                plot(time,SdX2e(:,3),'g')
                plot(time,SdX2e(:,4),'y')
                plot(time,SdX2e(:,5),'m')
                plot(time,SdX2e(:,6),'c')
                %plot(time,SdX2e(:,7),'k')
                legend('q1','q2','q3','q4','q5','q6','q7')
                plot(time,SddQ(:,1),'b','Linewidth',2)
                plot(time,SddQ(:,2),'r','Linewidth',2)
                plot(time,SddQ(:,3),'g','Linewidth',2)
                plot(time,SddQ(:,4),'y','Linewidth',2)
                plot(time,SddQ(:,5),'m','Linewidth',2)
                plot(time,SddQ(:,6),'c','Linewidth',2)
                %plot(time,SddQ(:,7),'k','Linewidth',2)
                plot(time,SddQd(:,1),'.r')


%title 'joint acceleration'

hold off
                titol=strcat('k11=',num2str(K11(1,1)),',k12=',num2str(K12(1,1)),',k21=',num2str(K21(1,1)),',k22=',num2str(K22(1,1)),',tau=',num2str(tau));
                title(titol)
                filename=['K11' num2str(K11(1,1)) '_K21' num2str(K21(1,1)) '_K12' num2str(K12(1,1)) '_K22' num2str(K22(1,1)) ',tau=' num2str(tau) '.fig'];
                saveas(hplot,filename)

end

toc