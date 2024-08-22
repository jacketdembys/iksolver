function bb=makefilter(n)
%figure(1);
f = [0 0.01 0.08 1]; a = [1 1 0 0];
b = firpm(n,f,a);
bb=b';
[h,w] = freqz(b,1,512);
plot(f,a,w/pi,abs(h))
legend('Ideal','firpm Design')
%paux=paux+2;