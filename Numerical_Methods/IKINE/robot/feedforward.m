
tic
t=0;

%close all
%WAM;
robot=wam;

n=robot.n;



tfinal=1;
niters=200;
h=tfinal/niters;


%% ROBOT INTEGRATION VARIABLES
ddq0=zeros(n,1);
ddq1=zeros(n,1);
ddq2=zeros(n,1);
ddq3=zeros(n,1);
dq1=zeros(n,1);
dq2=zeros(n,1);
dq3=zeros(n,1);



%% INITIALIZE STATE VARIABLES FALTA DEFINIR BE INICI!!
q0=ones(n,1)*0.2;%rand(n,1)%
qa0=q0;
qa1=q0;
qa2=q0;
qa3=q0;
qf=ones(n,1)*0.4;%4;%rand(n,1)%
%qf=[0.0938;    0.9150;    0.9298;   -0.6848;    0.9412;    0.9143%;   -0.0292];
dq0=zeros(n,1);
q=q0;
dq=dq0;
ddq=zeros(n,1);
uest=zeros(n,1);
[qd,dqd,ddqd] = jtraj(q0, qf, niters);
niters=niters*2;
for in=1:n
   qd(niters/2+1:niters,in)=qf(in,1)*ones(niters/2,1);
    
end

%qd=[qd;[qf(1,1)*ones(size(qd),1) qf(2,1)*ones(size(qd),1) qf(3,1)*ones(size(qd),1) qf(4,1)*ones(size(qd),1) qf(5,1)*ones(size(qd),1) qf(6,1)*ones(size(qd),1) qf(7,1)*ones(size(qd),1)]];%;qf(1,1)*ones(size(qd))];
dqd=[dqd;zeros(size(dqd))];%;zeros(size(dqd))];
ddqd=[ddqd;zeros(size(ddqd))];%;zeros(size(ddqd))];



uestold=uest;
dqold=dq;
qold=q;
ddqold=ddq;
fdmax=[10;0;0;0;0;0];

%% STORING VARIABLES
    SUE=[];
    SUD=[];
    SERR=[];
    SFD=[];
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
SX=[];
time=[];
SQE=[];


%% CONTROL PARAMETERS
%Kd=eye(n)*5;
%Kp=eye(n)*20;
Md=eye(n)*2;
control_signal_limit =[25 20 15 15 5 5 5];


%% PID parameters
Kyp=500*diag([2 3 1 1 0.1 0.5 0.1]);
Kyd=Kyp/2*0;
Kyi=eye(n)*0;


%% CONTROL VARIABLES
yc=zeros(n,1);


%% INITIAL CALCULATIONS        
eQold=zeros(n,1);
eQ=zeros(n,1);


for i=1:niters
    t0=t;
    t=i*tfinal/niters;
 


    %% ROBOT CONTROL 
%KINEMATICS AND DYNAMICS
    M=inertia(robot, q);
    if isnan(M(1,1))==1
       hola=1; 
        
    end
    N= coriolis(robot, q', dq')'+gravload(robot,q')';
    Md=inertia(robot, qd(i,:));
    Nd= coriolis(robot, qd(i,:), dqd(i,:))'+gravload(robot,qd(i,:))';
    J=jacob0(wam,q);
    T=fkine(wam,q);
    xe=T(1,4);

% MODEL part of controller (with added error)
eperc=0.0;
uc=M*ddqd(i,:)'+Nd;
%for i2=1:n
%   uc(i2)=uc(i2)*(1+(rand-0.5)*eperc);
%end
    
% PID CONTROLLER on joint position and velocity    
eQold2=eQold;
eQold=eQ;
eQ=qd(i,:)'-q;
%yc=yc+Kyp*(eQ-eQold)+Kyd/h*(eQ-2*eQold+eQold2)+h*Kyi*eQ;
yc=Kyp*eQ+0*Kyd/h*(eQ-eQold);

% commmand:
U=uc+yc;
    for iaux=1:n
        if abs(U(iaux))>control_signal_limit(iaux)
            U(iaux)=U(iaux)/abs(U(iaux))*control_signal_limit(iaux);
        end
    end




    %% ROBOT UPDATE DYNAMICS
    ddq0=ddq1;
    ddq1=ddq2;
    ddq2=ddq3;
    ddq3=ddqd(i,:)'+M\(Nd-N);%(U-N);
   if or(norm(ddq3)>100,isnan(ddq(1)))
     hola=1; 
    end
    dq0=dq1;
    dq1=dq2;
    dq2=dq3;
    dq3=integrator(ddq0,ddq1,ddq2,ddq3,h,10)+dq0;
    qa0=qa1;
    qa1=qa2;
    qa2=qa3;
    qa3=integrator(dq0,dq1,dq2,dq3,h,10)+qa0;
    %qold=q;
    %dqold=dq;
    ddqold=ddq;
    qold=qa2;
    dqold=dq2;
    if isnan(qa3(1,1))
     hola=1; 
    end
    q=qa3;

    dq=dq3;
    ddq=ddq3;
   


    %% SAVE VALUES
    %AUX=[AUX;(M*(dx2e-ddq))'];
    %SX1e=[SX1e;x1e'];
    %SX2e=[SX2e;x2e'];
%    SUE=[SUE;ue'];
%    SUD=[SUD;ud'];
%    SERR=[SERR;ue'-ud'];
%    SFD=[SFD;fd'];
%    SQE=[SQE;q'-qf'];
%    SX=[SX;xe];
    SQ=[SQ;q'];
    SQd=[SQd;qd(i,:)];
    SdQ=[SdQ;dq'];
    SdQd=[SdQd;dqd(i,:)];    
    SddQ=[SddQ;ddq'];
    SddQd=[SddQd;ddqd(i,:)];   
   
    if size(Suext)>0
         %Suext2=Suext;
        jaux=size(Suext,1);
        afilt=0.5;
        nolder=2;
        %Suext2=[Suext2;afilt*Suext(jaux,:)+(1-afilt)*Suext(jaux-1,:)];
        %uaux=afilt*Suext(jaux,:);
        uaux=bb(1)*Suext(jaux,:);
        for auxi=2:size(bb,1)
            if jaux-auxi+1>1
                uaux=uaux+bb(auxi)*Suext(jaux-auxi+1,:);
            end
        end 
        Suext2=[Suext2;uaux];
        %Suext2=[Suext2;afilt*Suext2(jaux-1,:)+(1-afilt)*Suext2(jaux-2,:)];
        %aixo funcionava i no se pq!!
    else
        Suext2=Suext;
    end
    
    
%    Sfext=[Sfext;fe'];
    time=[time;t];

end

T=1;
%spectral;
figure
plot(SQE)
hold on
plot()
titol=strcat('joint position error')
title(titol)

figure
plot(SQ)
hold on
plot(SQd,'Linewidth',2)
titol=strcat('q position')
title(titol)
legend('q1','q2','q3','q4','q5','q6','q7')


figure
plot(Sfext)
hold on
plot(SFD,'Linewidth',2)
titol=strcat('external wrench')
title('fe,fd')


figure
plot(SUE)
hold on
plot(SUD,'Linewidth',2)
titol=strcat('ue,ud')
title(titol)



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
toc