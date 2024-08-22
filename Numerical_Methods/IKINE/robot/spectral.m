T=1;
figure(1)
%plot(time,-De(:,j),'r')
hold on
for i=1:7
    if i==1
        col='b';
    elseif i==2
        
        col='g';
    elseif i==3
        
        col='r';
    elseif i==4
        
        col='c';
    elseif i==5
        
        col='m';
    elseif i==6
        
        col='y';
    elseif i==7
        col='k';
    end
    %N=size(time,1);
    if mod(N,2)==1
        N=N-1;  
    end
    f=Suext(1:N,i);
    p=abs(fft(f,N))/(N/2);
    p=p(1:(size(p,1))/2).^2;
    freq=[0:N/2-1]/T;
    semilogy(freq,p,'Color',col,'Linewidth',2);
end