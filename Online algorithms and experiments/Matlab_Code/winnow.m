% Function for winnow algorithm

function [w,theta] = winnow(x,y,alpha)

    [p,q] = size(x);
    %learning_rate = {5; 0.25; 0.03; 0.005; 0.001}; to be passed
    w = ones(1,q);
    theta = -q;
    
    for k = 1:20
        for i = 1:p
            temp = sign(dot(w,x(i,:)) + theta);
            if temp ~= y(i)
                w = w .* (alpha .^ ( y(i) * x(i,:) ));
            end
        end
    end

end