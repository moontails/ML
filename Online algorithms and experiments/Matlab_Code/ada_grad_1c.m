% Function for AdaGrad algorithm with margin

function [w] = ada_grad_1c(x,y,learning_rate)

    [p,q] = size(x);
    %learning_rate = {5; 0.25; 0.03; 0.005; 0.001}; to be passed
    w = zeros(1,q+1); %Adding theta to end of w
    x = [ x, ones(1,p)']; %X_n+1 = 1 to account for theta
    Gt = zeros(1, q+1);

    for i = 1:p
        temp = y(i)*dot(w,x(i,:));
        gt = -y(i)*x(i,:);
        Gt = Gt + gt.*gt;
        if temp <= 1
            for j = 1:q+1
                if Gt(j) == 0
                    continue
                end
                w(j) = w(j) - (learning_rate * gt(j) / sqrt(Gt(j)));
            end
        end
    end
    

end