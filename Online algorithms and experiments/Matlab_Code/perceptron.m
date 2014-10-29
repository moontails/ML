% Function for peceptron

function [w,theta] = perceptron(x,y)

    [p,q] = size(x);
    learning_rate = 1;
    w = zeros(1,q);
    theta = 0;
    
    for k = 1:20
        for i = 1:p
            temp = sign(dot(w,x(i,:)) + theta);
            if temp ~= y(i)
                w = w + (learning_rate*y(i)*x(i,:));
                theta = theta + (learning_rate*y(i));
            end
        end
    end
                
end