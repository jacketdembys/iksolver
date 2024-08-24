function displayPose(D_current, dimension)

    fprintf("\nCurrent pose:")
    for c=1:dimension
        if (c==1)
            fprintf("\nX = %f", D_current(c));
        elseif (c==2)            
            fprintf("\nY = %f", D_current(c));
        elseif (c==3)            
            fprintf("\nZ = %f", D_current(c));
        elseif (c==4)            
            fprintf("\nRoll = %f", D_current(c));
        elseif (c==5)            
            fprintf("\nPitch = %f", D_current(c));
        elseif (c==6)            
            fprintf("\nYaw = %f", D_current(c));
        end
    end
    fprintf("\n")

end