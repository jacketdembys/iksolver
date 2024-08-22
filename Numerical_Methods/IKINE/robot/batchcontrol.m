

clear all
WAM

for itau=1:3
    tau=itau*2;
    for i11=0:7
        for i12=0:2
            for i21=0:2
                for i22=0:7
                    [itau i11 i12 i21 i22]
                    K11=10*i11*eye(7);
                    K12=5*i12*eye(7);
                    K21=5*i21*eye(7);
                    K22=10*i22*eye(7);
                    scriptcontrol
                end
            end
        end
    end   
end