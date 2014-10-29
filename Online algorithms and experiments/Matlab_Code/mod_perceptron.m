% Function for peceptron

function [w,theta] = mod_perceptron(x,y,n)

    [p,q] = size(x);
    learning_rate = 1;
    w = zeros(1,q);
    theta = 0;
    for k = 1:20
        N = n*q;
        for i = 1:p
            temp = sign(dot(w,x(i,:)) + theta);
            if temp ~= y(i)
                if N == 0
                    w = w + (learning_rate*y(i)*x(i,:));
                    theta = theta + (learning_rate*y(i));
                else
                    N = N - 1;                    
                end
            end
        end
    end
                
end