tic
t=0;
n=6;
%close all
%WAM;
%robot=wam;
robot=p560;
tfinal=1;
niters=1000;
h=tfinal/niters


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


%PROXYS
%p0=dqd(1,:)';
%p1=dqd(1,:)';
%p2=dqd(1,:)';
%p3=dqd(1,:)';
%dp0=ddqd(1,:)';
%dp1=ddqd(1,:)';
%dp2=ddqd(1,:)';
%dp3=ddqd(1,:)';
K11=30;
K22=30;
K12=0;
K21=0;
Kd=30*eye(n);
Kp=100;

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

for i=1:niters
    t0=t;
    t=i*tfinal/niters;
    fext=uextgenerator(t,n)';
    n=robot.n;
    M=inertia(robot, q);
    %Kp=M;

    y=ddqd(i,:)'+Kd*(dqd(i,:)'-dq)+Kp*(qd(i,:)'-q);
    

    %tic
    %sim('robotsim.mdl')
   
    N= coriolis(robot, q', dq')'+gravload(robot,q')';
    uc=M*y+uest+N;
    
    ddq0=ddq1;
    ddq1=ddq2;
    ddq2=ddq3;
    ddq3=M\(uc-fext-N);
    
    
    %PROXYS
    %dp0=dp1;
    %dp1=dp2;
    %dp2=dp3;
    %dp3=ddqd(i,:)';
    %p0=dp1;
   % p1=dp2;
    %P2=dp3;
    %p3=integrator(dp0,dp1,dp2,dp3,h,i)+p0;
    
    dq0=dq1;
    dq1=dq2;
    dq2=dq3;
    dq3=integrator(ddq0,ddq1,ddq2,ddq3,h,i)+dq0;

    qa0=qa1;
    qa1=qa2;
    qa2=qa3;
    qa3=integrator(dq0,dq1,dq2,dq3,h,i)+qa0;
    q=qa3;
    dq=dq3;
    ddq=ddq3;
   
    %toc
    
    %OBSERVER
    out=observerest(robot,[uc;q;dq;x1e;x2e;dx2e;N],K11,K12,K21,K22);
    %APPLY LOW-PASS FILTER
    tau=1/(3*pi);
    a=h/(tau+h);
    
    dx1e=a*out(1:n)+(1-a)*dx1eold;
    dx2e=a*out(n+1:2*n)+(1-a)*dx2eold;
    uest=a*out(2*n+1:3*n)+(1-a)*uestold;

    
    dx1eold=dx1e;
    dx2eold=dx2e;
    uestold=uest;
    
    
    
    dy0=dy1;
    dy1=dy2;
    dy2=dy3;
    dy3=[dx1e;dx2e];
    xe=integrator(dy0,dy1,dy2,dy3,h,i)+[x1e;x2e];
    x1e=xe(1:n);
    x2e=xe(n+1:2*n);
    

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
    
    holaplot=0;
    if holaplot==1
        
        figure(3)
        hold on
        plot(t,uest(1),'b')
        plot(t,fext(1),'r')
        
        figure(4)
        hold on
        plot(t,q(1),'b')
        plot(t,qd(i,1),'r')
        
        figure(5)
        hold on
        plot(t,dq(1),'b')
        plot(t,dqd(i,1),'r')

        figure(6)
        hold on
        plot(t,y(1),'g')
        plot(t,ddq(1),'b')
        plot(t,ddqd(i,1),'r')
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
%close all;
figure(iaux)
                h=plot(time,SdX2e(:,1),'b')
                hold on
                plot(time,SdX2e(:,2),'r')
                plot(time,SdX2e(:,3),'g')
                plot(time,SdX2e(:,4),'y')
                plot(time,SdX2e(:,5),'m')
                plot(time,SdX2e(:,6),'c')
                %plot(time,SdX2e(:,7),'k')
                legend('q1','q2','q3','q4','q5','q6')%,'q7')
                plot(time,SddQ(:,1),'b','Linewidth',2)
                plot(time,SddQ(:,2),'r','Linewidth',2)
                plot(time,SddQ(:,3),'g','Linewidth',2)
                plot(time,SddQ(:,4),'y','Linewidth',2)
                plot(time,SddQ(:,5),'m','Linewidth',2)
                plot(time,SddQ(:,6),'c','Linewidth',2)
%                plot(time,SddQ(:,7),'k','Linewidth',2)
                plot(time,SddQd(:,1),'.r')


%title 'joint acceleration'

hold off
                titol=strcat('k11=',num2str(K11(1,1)),',k12=',num2str(K12(1,1)),',k21=',num2str(K21(1,1)),',k22=',num2str(K22(1,1)),',tau=',num2str(tau));
                title(titol)
                filename=['K11' num2str(K11(1,1)) '_K21' num2str(K21(1,1)) '_K12' num2str(K12(1,1)) '_K22' num2str(K22(1,1)) ',tau=' num2str(tau) '.fig'];
                saveas(h,filename)



toc