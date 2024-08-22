function q = pi2piD(q,Qlim)
%Qlim=[-ones(4,1) ones(4,1)]*pi/2;
for i=1:size(Qlim,1)
   q(i)=min(q(i),Qlim(i,2)-0.0000001);
   q(i)=max(q(i),Qlim(i,1)+0.0000001);
    
end