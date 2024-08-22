

q0=[-0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3]'+rand(7,1)*0.0001;
dq=ones(7,1);
WAM;
Tstore=[];
Qstore=[];
Cstore=[];
Sstore=[];
Kstore=[];
nu=10;
smin=0.05;


for i=0:10000
    q=q0+dq*i/10000;
    J=jacob0(wam,q);
    v=null(J);
    [uu,S,vv]=svd(J);
    S2=zeros(7,6);
    for j=1:6
        s=S(j);
        s2=(((s+nu)*s+2)*s+2*smin)/((s+nu)*s+2);
        S2(j,j)=1/s2;
    end    

    Jp=vv*S2*uu';
    W=[Jp v];
    Ki=diag([5 5 5 2 2 2 1]);
    KK=W*Ki*inv(W);
    e=ones(7,1)*0.001;%(rand(7,1)-ones(7,1)*0.5)*0.001;
    
    
    
    
    if cond(J)>6000
       hola=1; 
        
    end
    
    Tstore=[Tstore;(KK*e)'];
    Qstore=[Qstore;q'];
    Cstore=[Cstore;cond(J)];
    Sstore=[Sstore;S'];
    Kstore=[Kstore;v'];
    
end
figure
plot(Tstore)





