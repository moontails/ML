% Function for peceptron with margin

function [w,theta] = perceptron_margin(x,y,learning_rate)

    [p,q] = size(x);
    %learning_rate = {5; 0.25; 0.03; 0.005; 0.001}; to be passed
    margin_parameter = 1;
    w = zeros(1,q);
    theta = 0;
    
    for k = 1:20
        for i = 1:p
            temp = y(i)*(dot(w,x(i,:)) + theta);
            if temp <= margin_parameter
                w = w + (learning_rate*y(i)*x(i,:));
                theta = theta + (learning_rate*y(i));
            end
        end
    end

end