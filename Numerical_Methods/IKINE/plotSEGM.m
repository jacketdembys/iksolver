function plotSEGM(P0,P1,color,L)
t=0:0.1:1;
x=P0(1)+t*(P1(1)-P0(1));
y=P0(2)+t*(P1(2)-P0(2));
z=P0(3)+t*(P1(3)-P0(3));
plot3(x,y,z,'Linewidth',L,'Color',color);