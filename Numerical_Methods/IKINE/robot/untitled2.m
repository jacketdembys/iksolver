N11=10;
N22=10;
N12=5;
N21=5;

I6=eye(6);

for i11=0:N11
K11=i11*10;
    for i12=1:N12
        K12=i12*(-5)+10;
        for i21=1:N21
            K21=i21*5-10;
            for i22=1:N22
            K22=i22*10;
            K=[K11(1,1) K12(1,1);K21(1,1) K22(1,1)]
                scriptcontrol4;
                close all

                
            end
        end
    end
end
    