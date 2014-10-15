% Function for winnow algorithm with margin

function [count] = winnow_margin_2(x,y,alpha,margin_parameter)

    [p,q] = size(x);
    %learning_rate = {5; 0.25; 0.03; 0.005; 0.001}; to be passed
    w = ones(1,q);
    theta = -q;
    R = 1000;
    count = 0;
    while R ~= 0        
        for i = 1:p
            temp1 = y(i)*(dot(w,x(i,:)) + theta);
            temp = sign(dot(w,x(i,:)) + theta);
            if temp1 <= margin_parameter
                w = w .* (alpha .^ ( y(i) * x(i,:) ));                
            end
            if temp ~= y(i)
                R = 1000;
                count = count + 1;
            else
                R = R - 1;
                if R == 0
                    break
                end
            end
        end
    end
end