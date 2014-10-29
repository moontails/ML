% Function for peceptron with margin

function [count] = perceptron_margin_2(x,y,learning_rate)

    [p,q] = size(x);
    %learning_rate = {5; 0.25; 0.03; 0.005; 0.001}; to be passed
    margin_parameter = 1;
    w = zeros(1,q);
    theta = 0;
    R = 1000;
    count = 0;
    while R ~= 0    
        for i = 1:p
            temp1 = y(i)*(dot(w,x(i,:)) + theta);
            temp = sign(dot(w,x(i,:)) + theta);
            if temp1 <= margin_parameter
                w = w + (learning_rate*y(i)*x(i,:));
                theta = theta + (learning_rate*y(i));
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