%clear all
%% VARIABLES DEFINITION
tic
t=0;
n=7;
tfinal=1;
niters=500;
h=tfinal/niters;
ratiof=10;

%WAM;
robot=wam;
paux=5;
bb=makefilter(21);
n=wam.n;
K11=eye(n)*20;
Kf=eye(n);
K22=eye(n)*20;
K12=-5*eye(n);
K21=5*eye(n);
A=eye(n)-h*K11-h*(eye(n)-K12)*inv(eye(n)+K22)*K21;
B=[h*(K11+(eye(n)-K12)*inv(eye(n)+K22)*K21) h*eye(n)];
C=-inv(eye(n)+K22)*K21;
D=[inv(eye(n)+K22)*K21 eye(n)];





%% INITIAL CONDITIONS
q0=ones(n,1)*0.2;%rand(n,1)%
qf=ones(n,1)*0.4;%4;%rand(n,1)%
dq0=zeros(n,1);

q=q0;
qold=q;
dq=dq0;
dqold=dq0;
ddq=zeros(n,1);
ddqold=ddq;

x1=q0;
x1next=x1;
dx1=zeros(n,1);
x2old=dq0;
x2=dq0;
dx2old=zeros(n,1);

uest=zeros(n,1);
uestold=uest;
uaux=zeros(1,n);
ucold=zeros(n,1);

M=inertia(robot, q);
N= coriolis(robot, q', dq')'+gravload(robot,q')';
uc=M*ddq+N+0*uest;


%% GENERATE TRAJECTORY
[qd,dqd,ddqd] = jtraj(q0, qf, niters);
niters=niters*3;
qd=[qd;qf(1,1)*ones(size(qd));qf(1,1)*ones(size(qd))];
dqd=[dqd;zeros(size(dqd));zeros(size(dqd))];
ddqd=[ddqd;zeros(size(ddqd));zeros(size(ddqd))];



%% STORING VARIABLES
AUX=[];
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
Suext2=[];
Sfext=[];
time=[];
time2=[];




                    
for i=1:niters
    %% time indicators
    t0=t;
    t=i*tfinal/niters;
    
    %% current state keeping
    ucold2=ucold;
    ucold=uc;
    qold=q;
    dqold=dq;
    Nold=N;
    Mold=inertia(wam,qold);
    
    %% Current state updating
    fext=uextgenerator(t,7)';
    q=qd(i,:)';
    J=jacob0(robot,q);
    dq=dqd(i,:)';
    ddq=ddqd(i,:)';
    N=coriolis(robot, q', dq')'+gravload(robot,q')';
    uc=rne(wam,q',dq',ddq')'-fext;
   
    
    
    %% OBSERVER
     
    x1=x1next;
    x1next=A*x1+B*[q;dq];

    x2old2=x2old;
    x2old=x2;
    x2=C*x1+D*[q;dq];

    dx2old2=dx2old;
    dx2old=(x2-x2old)/h;
    
    SdX2e=[SdX2e;dx2old'];
    
    if mod(i,ratiof)==0 
        uestold=Mold*(Kf*(dqold-x2old)+dx2old)-ucold+Nold;
        time2=[time2;t];
        Suext=[Suext;uestold'];
    end 


    %% SAVE VALUES
%    AUX=[AUX;(M*(dx2e-ddq))'];
    SdX1e(i,:)=(x1next-x1)'/h;
    SX1e=[SX1e;x1'];
    SX2e=[SX2e;x2'];
    SQ=[SQ;q'];
    SQd=[SQd;qd(i,:)];
    SdQ=[SdQ;dq'];
    SdQd=[SdQd;dqd(i,:)];    
    SddQ=[SddQ;ddq'];
    SddQd=[SddQd;ddqd(i,:)];   
   
    applyPMfilter=0;
    if and(size(Suext)>0,applyPMfilter==1)
        jaux=size(Suext,1);
        afilt=0.5;
        uaux=bb(1)*Suext(jaux,:);
        
        for auxi=2:size(bb,1)
            if jaux-auxi+1>1
                uaux=uaux+bb(auxi)*Suext(jaux-auxi+1,:);
            end
        end 
        Suext2=[Suext2;uaux];

    else
        Suext2=Suext;
    end
    
    
    %% PLOTS
    Sfext=[Sfext;fext'];
    time=[time;t];

    
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
plotaltres=1;
plotaltres2=1;
if plotaltres==1
figure(paux)
               
                plot(time,SddQ,'Linewidth',2)
                hplot=plot(SdX2e(:,1),'b');
                
                hold on
                 %plot(time(1:size(time,1)-casos),SdX2e(:,1),'b')
                plot(SdX2e(:,2),'r')
                plot(SdX2e(:,3),'g')
                plot(SdX2e(:,4),'y')
                plot(SdX2e(:,5),'m')
                plot(SdX2e(:,6),'c')
                plot(SdX2e(:,7),'k')
                %legend('q1','q2','q3','q4','q5','q6')
                plot(SddQ(:,1),'b','Linewidth',2)
                plot(SddQ(:,2),'r','Linewidth',2)
                plot(SddQ(:,3),'g','Linewidth',2)
                plot(SddQ(:,4),'y','Linewidth',2)
                plot(SddQ(:,5),'m','Linewidth',2)
                plot(SddQ(:,6),'c','Linewidth',2)
                plot(SddQ(:,7),'k','Linewidth',2)
                plot(SddQd(:,1),'.r')
                title 'acceleration'
                titol=strcat('k11=',num2str(K11(1,1)),',k12=',num2str(K12(1,1)),',k21=',num2str(K21(1,1)),',k22=',num2str(K22(1,1)));
                title(titol)
                filename=['ACCK11' num2str(K11(1,1)) '_K21' num2str(K21(1,1)) '_K12' num2str(K12(1,1)) '_K22' num2str(K22(1,1)) '.fig'];
                saveas(hplot,filename)
  
                
                figure(paux+1)              
%                plot(time(1:size(time,1)-1),Suext)

                %plot(time,SddQ,'Linewidth',2)
                hplot2=plot(Suext2(:,1),'b');
                
                hold on
                 %plot(time(1:size(time,1)),SdX2e(:,1),'b')
                plot(Suext2(:,2),'r')
                plot(Suext2(:,3),'g')
                plot(Suext2(:,4),'y')
                plot(Suext2(:,5),'m')
                plot(Suext2(:,6),'c')
                plot(Suext2(:,7),'k')
                %legend('q1','q2','q3','q4','q5','q6')
                titol=strcat('k11=',num2str(K11(1,1)),',k12=',num2str(K12(1,1)),',k21=',num2str(K21(1,1)),',k22=',num2str(K22(1,1)));
%plot(time(1:size(time,1)-casos),Suext)
                
                title(titol)
                filename=['UestK11' num2str(K11(1,1)) '_K21' num2str(K21(1,1)) '_K12' num2str(K12(1,1)) '_K22' num2str(K22(1,1)) '.fig'];
                saveas(hplot2,filename)


%                plot(time,Sfext,'Linewidth',2)
        

                
                if plotaltres2==1

                 figure(paux+2)
                                hplot=plot(time,SX2e(:,1),'b');
                                hold on
                                plot(time,SX2e(:,2),'r')
                                plot(time,SX2e(:,3),'g')
                                plot(time,SX2e(:,4),'y')
                                plot(time,SX2e(:,5),'m')
                                plot(time,SX2e(:,6),'c')
                                plot(time,SX2e(:,7),'k')
                                legend('q1','q2','q3','q4','q5','q6','q7')
                                plot(time,SdQ(:,1),'b','Linewidth',2)
                                plot(time,SdQ(:,2),'r','Linewidth',2)
                                plot(time,SdQ(:,3),'g','Linewidth',2)
                                plot(time,SdQ(:,4),'y','Linewidth',2)
                                plot(time,SdQ(:,5),'m','Linewidth',2)
                                plot(time,SdQ(:,6),'c','Linewidth',2)
                                plot(time,SdQ(:,7),'k','Linewidth',2)
                                %plot(time,SdQd(:,1),'.r')
                                title 'velocity'
                 figure(paux+3)
                                hplot=plot(time,SX1e(:,1),'b');
                                hold on
                                plot(time,SX1e(:,2),'r')
                                plot(time,SX1e(:,3),'g')
                                plot(time,SX1e(:,4),'y')
                                plot(time,SX1e(:,5),'m')
                                plot(time,SX1e(:,6),'c')
                                plot(time,SX1e(:,7),'k')
                                legend('q1','q2','q3','q4','q5','q6','q7')
                                plot(time,SQ(:,1),'b','Linewidth',2)
                                plot(time,SQ(:,2),'r','Linewidth',2)
                                plot(time,SQ(:,3),'g','Linewidth',2)
                                plot(time,SQ(:,4),'y','Linewidth',2)
                                plot(time,SQ(:,5),'m','Linewidth',2)
                                plot(time,SQ(:,6),'c','Linewidth',2)
                                plot(time,SQ(:,7),'k','Linewidth',2)
                                %plot(time,SQd(:,1),'.r')
                                title 'position'
%title 'joint acceleration'
                end

hold off
                %titol=strcat('k11=',num2str(K11(1,1)),',k12=',num2str(K12(1,1)),',k21=',num2str(K21(1,1)),',k22=',num2str(K22(1,1)),',tau=',num2str(tau));
                %title(titol)
                %filename=['K11' num2str(K11(1,1)) '_K21' num2str(K21(1,1)) '_K12' num2str(K12(1,1)) '_K22' num2str(K22(1,1)) ',tau=' num2str(tau) '.fig'];
                %saveas(hplot,filename)

end

toc

%BE=[]
%for i=1:niters
%    BE=[BE;(inertia(wam,SQ(i,:))*(SdX2e(i,:)'-SddQ(i,:)'))'];  
%end