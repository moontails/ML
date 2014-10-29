% Function for AdaGrad algorithm with margin

function [count] = ada_grad_2(x,y,learning_rate)

    [p,q] = size(x);
    %learning_rate = {5; 0.25; 0.03; 0.005; 0.001}; to be passed
    w = zeros(1,q+1); %Adding theta to end of w
    x = [ x, ones(1,p)']; %X_n+1 = 1 to account for theta
    Gt = zeros(1, q+1);
    R = 1000;
    count = 0;
    while R ~= 0
        for i = 1:p
            temp1 = y(i)*dot(w,x(i,:));
            temp = sign(dot(w,x(i,:)));
            gt = -y(i)*x(i,:);
            Gt = Gt + gt.*gt;
            if temp1 <= 1
                for j = 1:q+1
                    if Gt(j) == 0
                        continue
                    end
                    w(j) = w(j) - (learning_rate * gt(j) / sqrt(Gt(j)));
                end
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