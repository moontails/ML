% Configuration 3
l = 10;
m = 1000; 
n = 1000;

% Generate noisy data of 50000 examples and D1 and D2
[y,x] = gen(l,m,n,50000,1);
[T_y,T_x] = gen(l,m,n,10000,0);

% Perceptron
% Tuning Step
% No Paramter to tune
% So Directly Training using 100% noisy data
[w, theta ] = perceptron(x,y); 
count = 0;
% Evaluation - Reporting the Error Rate
for i = 1:10000
    temp = sign(dot(w,T_x(i,:)) + theta);
    if temp ~= T_y(i)
        count = count + 1;
    end
end
display(count/10000)

% Performed tuning and found out the best parameter

% Perceptron with margin
learning_rate = 0.03;
% Training on 100% noisy data
[w,theta] = perceptron_margin(x,y,learning_rate);
count = 0;
% Evaluation
for i = 1:10000
    temp = sign(dot(w,T_x(i,:)) + theta);
    if temp ~= T_y(i)
        count = count + 1;
    end
end
display(count/10000)

% Winnow
alpha = 1.1;
% Training on 100% noisy data
[w,theta] = winnow(x,y,alpha);
count = 0;
% Evaluation
for i = 1:10000
    temp = sign(dot(w,T_x(i,:)) + theta);
    if temp ~= T_y(i)
        count = count + 1;
    end
end
display(count/10000)

% Winnow with margin
alpha = 1.1;
margin_parameter = 0.006;
% Training on 100% noisy data
[w,theta] = winnow_margin(x,y,alpha,margin_parameter);
count = 0;
% Evaluation - Error Rate
for i = 1:10000
    temp = sign(dot(w,T_x(i,:)) + theta);
    if temp ~= T_y(i)
        count = count + 1;
    end
end
display(count/10000)

% AdaGrad
learning_rate = 1.5;
% Training on 100% noisy data
T_x = [ T_x, ones(1,10000)'];
[w] = ada_grad(x,y,learning_rate);
count = 0;
% Evaluation - Error Rate
for i = 1:10000
    temp = sign(dot(w,T_x(i,:)));
    if temp ~= T_y(i)
        count = count + 1;
    end
end
display(count/10000)
