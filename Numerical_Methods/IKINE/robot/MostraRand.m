
function q=MostraRand
Qlim=zeros(7,2);
Qlim(1,1)=-2.6;
Qlim(1,2)=2.6;
Qlim(2,1)=-2.0;
Qlim(2,2)=2.0;
Qlim(3,1)=-2.8;
Qlim(3,2)=2.8;
Qlim(4,1)=-0.9;
Qlim(4,2)=3.1;
Qlim(5,1)=-4.8;
Qlim(5,2)=1.3;
Qlim(6,1)=-1.6;
Qlim(6,2)=1.6;
Qlim(7,1)=-2.2;
Qlim(7,2)=2.2;


aux1=rand(1);
q(1,1)=Qlim(1,1)+(Qlim(1,2)-Qlim(1,1))*aux1;
aux2=rand(1);
q(2,1)=Qlim(2,1)+(Qlim(2,2)-Qlim(2,1))*aux2;  
aux3=rand(1);
q(3,1)=Qlim(3,1)+(Qlim(3,2)-Qlim(3,1))*aux3;
aux4=rand(1);
q(4,1)=Qlim(4,1)+(Qlim(4,2)-Qlim(4,1))*aux4;
aux5=rand(1);
q(5,1)=Qlim(5,1)+(Qlim(5,2)-Qlim(5,1))*aux5;
aux6=rand(1);
q(6,1)=Qlim(6,1)+(Qlim(6,2)-Qlim(6,1))*aux6;
aux7=rand(1);
q(7,1)=Qlim(7,1)+(Qlim(7,2)-Qlim(7,1))*aux7;