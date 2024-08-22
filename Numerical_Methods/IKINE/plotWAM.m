function [P0,P1,P2,P3,P4,P5]=plotWAM(q,qlor,L)
    
    w2=WAMarm2;
    w4=WAMarm4;
    w7=WAMarm7;
    T2=fkine(w2,q(1:2));
    T4=fkine(w4,q(1:4));
    T7=fkine(w7,q(1:7));
    
    
    P0=[0;0;0];
    P1=T2*[0;0;0.55;1];
    P2=T4*[0.045;0;0;1];
    P3=T4(1:3,4);
    P4=T7(1:3,4)-T7(1:3,3)*0.06;
    P5=T7(1:3,4);

    plotSEGM(P0(1:3),P1(1:3),qlor,L)
    hold on
    plotSEGM(P1(1:3),P2(1:3),qlor,L)
    plotSEGM(P2(1:3),P3(1:3),qlor,L)
    plotSEGM(P3(1:3),P4(1:3),qlor,L)
    plotSEGM(P4(1:3),P5(1:3),qlor,L)
    
    plot3(P1(1),P1(2),P1(3),'.','Color',qlor);
    plot3(P2(1),P2(2),P2(3),'.','Color',qlor);
    plot3(P3(1),P3(2),P3(3),'.','Color',qlor);
    plot3(P4(1),P4(2),P4(3),'.','Color',qlor);
    plot3(P5(1),P5(2),P5(3),'.','Color',qlor);
