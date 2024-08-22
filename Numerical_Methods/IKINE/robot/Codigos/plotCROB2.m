
load Kp1000Kd100.mat;
aux=100;
plot(time,q(:,index),'color',[0,0,(1000-aux)/1000]);
hold on;
load Kp1000Kd250.mat;
aux=250;
plot(time,q(:,index),'color',[0,0,(1000-aux)/1000]);
load Kp1000Kd500.mat;
aux=500;
plot(time,q(:,index),'color',[0,0,(1000-aux)/1000]);
load Kp1000Kd1000.mat;
aux=1000;
plot(time,q(:,index),'color',[0,0,(1000-aux)/1000]);

load Kp1k_312111_Kd100.mat;
aux=100;
plot(time,q(:,index),'color',[(1000-aux)/1000,(1000-aux)/1000,0]);
hold on;
load Kp1k_312111_Kd250.mat;
aux=250;
plot(time,q(:,index),'color',[(1000-aux)/1000,(1000-aux)/1000,0]);
load Kp1k_312111_Kd500.mat;
aux=500;
plot(time,q(:,index),'color',[(1000-aux)/1000,(1000-aux)/1000,0]);
load Kp1k_312111_Kd1000.mat;
aux=1000;
plot(time,q(:,index),'color',[(1000-aux)/1000,(1000-aux)/1000,0]);


legend('kd100','kd250','kd500','kd1000','312111_kd100','312111_kd200','312111_kd500','312111_kd1000')