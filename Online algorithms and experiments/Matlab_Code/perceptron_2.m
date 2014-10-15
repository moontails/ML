% Function for peceptron

function [count] = perceptron_2(x,y)

    [p,q] = size(x);
    learning_rate = 1;
    w = zeros(1,q);
    theta = 0;
    R = 1000;
    count = 0;
    while R ~= 0
        for i = 1:p
            temp = sign(dot(w,x(i,:)) + theta);
            if temp ~= y(i)
                R = 1000;
                w = w + (learning_rate*y(i)*x(i,:));
                theta = theta + (learning_rate*y(i));
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