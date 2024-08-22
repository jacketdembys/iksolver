
P=zeros(12,1);
NN=100000
count=0;
for i=1:NN
    roll=ceil(6*rand(3,1));
    roll2=sort(roll);
    V=roll2(3)+roll2(2);
    P(V)=P(V)+1;
    
    rollb=ceil(6*rand(2,1));
    rollb2=sort(rollb);
    Vb=rollb2(1)+rollb2(2);
    
    if V>=Vb
       count=count+1; 
    end
        
end
Pwin=count/NN;
P2=P/NN;
    %rollb=ceil(6*rand(3,1));
    %R1=max(roll);
    %R3=min(roll);
