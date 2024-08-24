function ss = checkIfSingularityHappened(J, Q_current, i)

    if length(Q_current) == 6
        %fprintf("\nCurrent Determinant J:")
        det_J = det(J); %#ok<NOPRT>

        if (det_J == 0)
            %fprintf("Singular position at iteration: [%f]", (i));
            ss = i;
        end

    else

        %fprintf("\nCurrent Determinant J*J':")
        %det_J = det(J*J') %#ok<NOPTS>
        rJ = rank(J); %#ok<NOPRT>


        if (rJ < min(6, length(Q_current)))
            %fprintf("Singular position at iteration: [%f]", (i));
            ss = i;
        else
            ss = 0;
        end

    end

end