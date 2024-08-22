close all
clear all
WAM;
M=csvread('ForcelogOutput_6.txt');
bb=makefilter(15);
time=M(:,1);
q=M(:,2:8);
p=M(:,9:15);
dq=M(:,16:22);
v=M(:,23:29);
ddqd=M(:,30:36);
a=M(:,37:43);
uc=M(:,44:50);
uest=M(:,51:57);
fest=[];
uestfiltered=[];
for i=1:size(uest,1)
    if i>=size(bb,1)
        uaux=bb(1)*uest(i,:);
        for auxi=2:size(bb,1)
            if i-auxi+1>1
                uaux=uaux+bb(auxi)*uest(i-auxi+1,:);
            end
        end 
        uestfiltered=[uestfiltered;uaux];

    else
        uaux=uest(i,:);
        uestfiltered=[uestfiltered;uaux];
    end
    
    J=jacob0(wam,q(i,:));
    fest=[fest;(pinv(J')*uaux')'];
end

joint_val=2

figure
plot(time,[q(:,joint_val),p(:,joint_val)])
title('Position')
legend('q','p');

figure
plot(time,[dq(:,joint_val),v(:,joint_val)])
title('Velocity')
legend('dq','v');


figure
plot(time,[ddqd(:,joint_val),a(:,joint_val)])
title('Acceleration')
legend('ddqd','a');


figure
plot(time,uest(:,joint_val))
hold on
plot(time,uestfiltered(:,joint_val),'LineWidth',2,'Color','r')
title('Estimated external joint torque')
legend('non-filtered','filtered')
hold off

figure
plot(time,fest)
title('End effector estimated wrench')
legend('Fx','Fy','Fz','Mx','My','Mz')
