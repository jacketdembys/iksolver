
tic
t=0;
n=7;
%close all
%WAM;
%robot=wam;

%robot=perturb(p560,pro);
robot=wam;
rexact=wam;
tfinal=1;
niters=500;
h=tfinal/niters;


dy0=zeros(2*n,1);
dy1=zeros(2*n,1);
dy2=zeros(2*n,1);
dy3=zeros(2*n,1);

ddq0=zeros(n,1);
ddq1=zeros(n,1);
ddq2=zeros(n,1);
ddq3=zeros(n,1);

dq1=zeros(n,1);
dq2=zeros(n,1);
dq3=zeros(n,1);

q0=ones(n,1)*0.2;%rand(n,1)%
qa0=q0;
qa1=q0;
qa2=q0;
qa3=q0;
qf=ones(n,1)*0.4%4;%rand(n,1)%

%q0=zeros(n,1);
x1e=q0;
dx1e=zeros(n,1);
dq0=zeros(n,1);
dqold=dq0;
xe0=[q0;dq0];
xe1=zeros(2*n,1);
xe2=zeros(2*n,1);
xe3=zeros(2*n,1);

x2old=dq0;
x2e=dq0;
dx2e=zeros(n,1);
q=q0;
dq=dq0;
ddq=zeros(n,1);
uest=zeros(n,1);
[qd,dqd,ddqd] = jtraj(q0, qf, niters);
niters=niters*3;
qd=[qd;qf(1,1)*ones(size(qd));qf(1,1)*ones(size(qd))];
dqd=[dqd;zeros(size(dqd));zeros(size(dqd))];
ddqd=[ddqd;zeros(size(ddqd));zeros(size(ddqd))];


dx1eold=dx1e;
dx2old=dx2e;
dx2old2=dx2e;
uestold=uest;

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
e1=zeros(n,1);
y=zeros(n,1);
x1next=x1e+h*(x2e+K11*(q-x1e)+K12*(dq-x2e));
x2next=2*x2e-x2old+h*(K21*(q-x1e)+K22*(dq-x2e)+dq2-x2old);
        
                %kp = diag([ 900  2500 600  500   50   50    8])/100;
                %Kp=diag([ 900  1000 600  500   50   50 8])/50;
Ki = diag([2.5    5   2 0.5  0.5  0.5  0.1]);
Kp=5*diag([10 10 15 5 5 5 2])
Kd =Kp.^0.5*2;% diag([10   20    5    2  0.5  0.5 0.05]);
%kd=Kd;
%kp=Kp;
control_signal_limit =[25 20 15 15 5 5 5]*2;
interror=0;
M=inertia(robot, q);
N= coriolis(robot, q', dq')'+gravload(robot,q')';
uc=M*y+N+0*uest;
dqold=dq;
qold=q;
ddqold=ddq;
                
         
A=eye(n)-h*K11-h*(eye(n)-K12)*inv(eye(n)+K22)*K21;
B=[h*(K11+(eye(n)-K12)*inv(eye(n)+K22)*K21) h*eye(n)];
C=-inv(eye(n)+K22)*K21;
D=[inv(eye(n)+K22)*K21 eye(n)];
                
fcut=0.002;
invtau=2*pi*fcut;   
a=h/(invtau+h);
uaux=zeros(1,n);
ucold=zeros(n,1);
            dx2old5=zeros(n,1);
            dx2old4=zeros(n,1);
            dx2old3=zeros(n,1);
            dx2old2=zeros(n,1);
ratiof=4;
maxjerk=100*h;
time2=[];

                    
for i=1:niters
    t0=t;
    t=i*tfinal/niters;
    ucold2=ucold;
    ucold=uc;
    fext=uextgenerator(t,7)';
    Nexact= coriolis(rexact, q', dq')'+gravload(rexact,q')';  


    %% ROBOT CONTROL 
    %if mod(i,ratiof)==0
        n=robot.n;
%        Mold2=Mold;
        Mold=M;
        M=inertia(robot, q);

%        Nold2=Nold;
        Nold=N;
        N= coriolis(robot, q', dq')'+gravload(robot,q')';
        y=ddqd(i,:)'+Kd*(dqd(i,:)'-dq)+Kp*(qd(i,:)'-q);
        for iaux=1:n
            if abs(y(iaux))>control_signal_limit(iaux)
                y(iaux)=y(iaux)/abs(y(iaux))*control_signal_limit(iaux);
            end
        end
        uc=M*y+N+(uaux');
    %end
    
    %% OBSERVER
  
    x1next=x1e+h*(x2e+K11*(q-x1e)+K12*(dq-x2e));
    SdX1e(i,:)=(x1next-x1e)'/h;
    
    
    
            x1old=x1e;
            x1e=A*x1old+B*[qold;dqold];
            if i>5
            end
            x2old2=x2old;
            x2old=x2e;
            x2e=C*x1e+D*[q;dq];
            %x2e=(1-a)*x2e+a*x2old;
            
    %dx2old=dx2e;
    %dx2e=(x2e-x2old)/h; %OLD
    %dx2e=ddq; %OJOOOO QUE FAIG TRAMPES!!
    %dx2eA=1/h^2*(x1next-2*x1e+x1old);
    %dx2e=(dx2eA+dx2eB)/2;
    %dx2e=(1-a)*dx2e+a*dx2old;
    %if i>1
    %    SdX2e(i-1,:)=dx2e'; %has a 1 period delay!!!
    %end
    %uestold=uest;
    %uest=M*(-Kf*(dqold-x2old)-dx2e)+ucold-Nold; %has a 1 period delay!!
    %uest=(1-a)*uest+a*uestold; %it is, in fact, the old one!
    
    %x1old=x1e;
    %x1e=A*x1old+B*[qold;dqold];
    %x2old=x2e;
    %x2e=C*x1e+D*[q;dq];
    tcasos=1;
    dx2old2=dx2old;

if tcasos==1
            casos=1;
            dx2old=(x2e-x2old)/h;
            %if i<0
            %    for kkk=1:n
            %        if abs(dx2old(kkk)-dx2old2(kkk))>maxjerk
            %            dx2old(kkk)=dx2old2(kkk)+(dx2old(kkk)-dx2old2(kkk))/norm(dx2old(kkk)-dx2old2(kkk))*maxjerk;
            %        end
            %    end
            %end
            SdX2e=[SdX2e;dx2old'];
            if mod(i,ratiof)==0 
                 uestold=Mold*(-Kf*(dqold-x2old)-dx2old)+ucold-Nold;
                 Suext=[Suext;uestold'];
                 time2=[time2;t];
            end
           

              
               

        elseif tcasos==-1
            casos=1;
            dx2old2=(x2e-x2old)/h;
            
            uestold2=Mold2*(-Kf*(dqold2-x2old2)-dx2old2)+ucold2-Nold2;
            %uest=M*(-Kf*(dq-x2e)-dx2e)+uc-N;
            if i>2
               SdX2e=[SdX2e;dx2old2'];
               Suext=[Suext;uestold2'];
               
               
                %SdX2e(i-1,:)=dx2old ; %has a 1 period delay!!!
               %Suext(i-1,:)=uestold;

            end

        elseif tcasos==0
            casos=0;
            dx2old2=dx2old;
            dx2old=(x2e-x2old)/h;
            dx2e=3*dx2old-2*dx2old2;
            %au=0.8;
            %dx2e=(1-a)*dx2e+a*dx2old;

            uestold=Mold*(-Kf*(dqold-x2old)-dx2old)+ucold-Nold;
            uest=M*(-Kf*(dq-x2e)-dx2e)+uc-N;
            %uest=(1-au)*uest+au*uestold;
            if i>1
               %SdX2e(i-1,:)=dx2e;%old'; %has a 1 period delay!!!
               %Suext(i-1,:)=uest;%old';
                Suext(i,:)=uest;
               SdX2e(i,:)=dx2e;
            end
        elseif tcasos==2
            casos=0;
            XX=[0;h;2*h;3*h;4*h];

            if i>5
                dx2old=(x2e-x2old)/h;
                SdX2e(i-1,:)=dx2old;
                YY=[SdX2e(i-5,:);SdX2e(i-4,:);SdX2e(i-3,:);SdX2e(i-2,:);SdX2e(i-1,:)];
                for jj=1:n
                     pj=polyfit(XX,YY(:,jj),1);
                     dx2e(jj,1)=polyval(pj,5*h);
                end


            else
                dx2old2=dx2old;
                dx2old=(x2e-x2old)/h;
                dx2e=3*dx2old-2*dx2old2; 
            end

            uest=M*(-Kf*(dq-x2e)-dx2e)+uc-N;
            if i>1
               %SdX2e(i-1,:)=dx2e;%old'; %has a 1 period delay!!!
               %Suext(i-1,:)=uest;%old';
                Suext(i,:)=uest;
                SdX2e(i,:)=dx2e;
            end
            
        elseif tcasos==3
            dx2oldc=zeros(n,1);
            dx2old5=dx2old4;
            dx2old4=dx2old3;
            dx2old3=dx2old2;
            dx2old2=dx2oldc;
            dx2old=(x2e-x2old)/h;
            
            for ifit=1:n
                pol=polyfit([0 h 2*h 3*h 4*h]',[dx2old5(ifit) dx2old4(ifit) dx2old3(ifit) dx2old2(ifit) dx2old(ifit)]',1);
                dx2oldc(ifit,1)=polyval(pol,5*h);
            
            end
             uestold=Mold*(-Kf*(dqold-x2old)-dx2oldc)+ucold-Nold;
            %uest=M*(-Kf*(dq-x2e)-dx2e)+uc-N;
            if i>1
               SdX2e=[SdX2e;dx2old'];
               Suext=[Suext;uestold'];
               
               
                %SdX2e(i-1,:)=dx2old ; %has a 1 period delay!!!
               %Suext(i-1,:)=uestold;

            end       
end
        
        


    %% ROBOT UPDATE DYNAMICS
    ddq0=ddq1;
    ddq1=ddq2;
    ddq2=ddq3;
    ddq3=M\(uc-fext-Nexact);
    dq0=dq1;
    dq1=dq2;
    dq2=dq3;
    dq3=integrator(ddq0,ddq1,ddq2,ddq3,h,10)+dq0;
    qa0=qa1;
    qa1=qa2;
    qa2=qa3;
    qa3=integrator(dq0,dq1,dq2,dq3,h,10)+qa0;
    qold=q;
    dqold2=dqold;
    dqold=dq;
    ddqold=ddq;
    
    qold=qa2;
    dqold=dq2;
    q=qa3;
    dq=dq3;
    ddq=ddq3;
   


    %% SAVE VALUES
    AUX=[AUX;(M*(dx2e-ddq))'];
    SX1e=[SX1e;x1e'];
    SX2e=[SX2e;x2e'];
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
    
    
    %% PLOTS
    Sfext=[Sfext;fext'];
    time=[time;t];
    
    holaplot=0;
%    iplots=0;
    if holaplot==1
        
        figure(iplots)
        hold on
        plot(t,uest(7),'b')
        plot(t,fext(7),'r')
        
        figure(iplots+1)
        hold on
        plot(t,q(7),'b')
        plot(t,qd(i,7),'r')
        plot(t,x1e(7,1),'k')
        
        figure(iplots+2)
        hold on
        plot(t,dq(7),'b')
        plot(t,dqd(i,7),'r')
        plot(t,SdX1e(i,7),'g')
        plot(t,x2e(7,1),'k')
        
        figure(iplots+3)
        hold on
        plot(t,y(7),'g')
        plot(t,ddq(7),'b')
        plot(t,ddqd(i,7),'r')
        plot(t,SdX2e(i,7),'k')
        
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