% Function for winnow algorithm

function [count] = winnow_2(x,y,alpha)

    [p,q] = size(x);
    %learning_rate = {5; 0.25; 0.03; 0.005; 0.001}; to be passed
    w = ones(1,q);
    theta = -q;
    R = 1000;
    count = 0;
    while R ~= 0
        for i = 1:p
            temp = sign(dot(w,x(i,:)) + theta);
            if temp ~= y(i)
                R = 1000;
                count = count + 1;
                w = w .* (alpha .^ ( y(i) * x(i,:) ));
            else
                R = R - 1;
                if R == 0
                    break
                end
            end
        end
    end
   

end