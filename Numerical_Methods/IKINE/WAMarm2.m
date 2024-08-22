function r=WAMarm2
%links alpha A theta D
L1=link([-pi/2 0 0 0 0],'standard');
L2=link([pi/2 0 0 0 0],'standard');
%L3=link([-pi/2 0.045 0 0.55 0],'standard');
%L4=link([pi/2 -0.045 0 0 0],'standard');

%L7=link([0 0 0 0 0],'standard'); 
%deixem el redundant a part
%r=robot({L1 L2 L3}) 
r=robot({L1 L2});  

%plot(r,[0,0,0,0,0,0,0])
%drivebot(r)
%q = ikine(r,T)
%q1=centratV(q,6)

%provaaa=fkine(r,[0.58178 0.81449 0.54299 0.54299 0.62832 0.65935])
%Mobj =
%
%   -0.8279   -0.5399    0.1519    0.5379
%    0.0744    0.1627    0.9839    0.4536
%   -0.5559    0.8259   -0.0946    0.4691
%         0         0         0    1.0000