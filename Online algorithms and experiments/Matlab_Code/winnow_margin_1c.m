% Function for winnow algorithm with margin

function [w,theta] = winnow_margin_1c(x,y,alpha,margin_parameter)

    [p,q] = size(x);
    %learning_rate = {5; 0.25; 0.03; 0.005; 0.001}; to be passed
    w = ones(1,q);
    theta = -q;
    
    
    for i = 1:p
        temp = y(i)*(dot(w,x(i,:)) + theta);
        if temp <= margin_parameter
            w = w .* (alpha .^ ( y(i) * x(i,:) ));                
        end
    end

end