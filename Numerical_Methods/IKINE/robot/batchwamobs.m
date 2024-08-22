WAM
robot=wam
n=robot.n
Kact=eye(7);
Kp=25*eye(7);
Kd=50*eye(7);

cutfreq=2*pi;
q0=ones(7,1)*0.2;
qf=ones(7,1)*0.5;
dq0=zeros(7,1);


                close all
                clear estimations
                clear simout
                clear time
                clear De
                K11=15*eye(7);
                K12=15*eye(7);
                K21=15*eye(7);
                K22=15*eye(7);
                conv=K11(1,1)+(1-K12(1,1))/(1+K22(1,1))*K21(1,1)
                sim('OBS7wamOP')
                
                De=estimations(:,7*n+1:8*n);
                time=estimations(:,8*n+1);
                
                titol=strcat('k11=',num2str(K11(1,1)),',k12=',num2str(K12(1,1)),',k21=',num2str(K21(1,1)),',k22=',num2str(K22(1,1)))
                figure(1)
                

                title(titol)

                h=plot(time,-De(:,1),'b')
                hold on
                plot(time,-De(:,2),'r')
                plot(time,-De(:,3),'g')
                plot(time,-De(:,4),'y')
                plot(time,-De(:,5),'m')
                plot(time,-De(:,6),'c')
                plot(time,-De(:,7),'k')
                legend('q1','q2','q3','q4','q5','q6','q7')
                hold off
                filename=['K11' num2str(K11(1,1)) '_K21' num2str(K21(1,1)) '_K12' num2str(K12(1,1)) '_K22' num2str(K22(1,1)) '.fig'];
                saveas(h,filename)










%qf =

%    0.0357
%    0.8491
%%    0.9340
 %   0.6787
 %%   0.7577
 %   0.7431
 %   0.3922

%q0=rand(7,1)

%q0 =

%    0.6555
%    0.1712
%    0.7060
%    0.0318
%%    0.2769
%%    0.0462
%    0.0971