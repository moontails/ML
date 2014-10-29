% Configuration 3
l = 10;
m = 1000; 
n = 1000;

% Generate noisy data of 50000 examples and D1 and D2
[y,x] = gen(l,m,n,50000,1);
[T_y,T_x] = gen(l,m,n,10000,0);

data = [x,y];
[p,q] = size(data);
rng(7);
D1 = datasample(data,5000);
D1_x = D1(:,1:q-1);
D1_y = D1(:,q);

D2 = datasample(data,5000);
D2_x = D2(:,1:q-1);
D2_y = D2(:,q);

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

% Perceptron with margin
learning_rate = [1.5, 0.25, 0.03, 0.005, 0.001];
count = [0, 0, 0, 0, 0];
% Tuning
for k = 1:5
    [w,theta] = perceptron_margin(D1_x,D1_y,learning_rate(k));
    for i = 1:5000
        temp = sign(dot(w,D2_x(i,:)) + theta);
        if temp == D2_y(i)
            count(k) = count(k) + 1;
        end
    end
end
% Training on 100% noisy data
[~,learning_ratei] = max(count);
learning_rate = learning_rate(learning_ratei);
display(learning_rate)
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
alpha = [1.1, 1.01, 1.005, 1.0005, 1.0001];
count = [0, 0, 0, 0, 0];
for k = 1:5
    [w,theta] = winnow(D1_x,D1_y,alpha(k));
    for i = 1:5000
        temp = sign(dot(w,D2_x(i,:)) + theta);
        if temp == D2_y(i)
            count(k) = count(k) + 1;
        end
    end
end
% Training on 100% noisy data
[~,alpha_i] = max(count);
alpha = alpha(alpha_i);
display(alpha)
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
alpha =[1.1, 1.01, 1.005, 1.0005, 1.0001];
margin_parameter = [2.0, 0.3, 0.04, 0.006, 0.001];
count = zeros(5,5);
for k = 1:5
    for l = 1:5
        [w,theta] = winnow_margin(D1_x,D1_y,alpha(k),margin_parameter(l));
        count(k,l) = 0;
        for i = 1:5000
            temp = sign(dot(w,D2_x(i,:)) + theta);
            if temp == D2_y(i)
                count(k,l) = count(k,l) + 1;
            end
        end
    end
end
% Training on 100% noisy data
[temp1,alpha_i] = max(count);
alpha_i = max(alpha_i);
[temp2, mp_i] = max(temp1);
alpha = alpha(alpha_i);
margin_parameter = margin_parameter(mp_i);
display(alpha)
display(margin_parameter)
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
learning_rate = [1.5, 0.25, 0.03, 0.005, 0.001];
D2_x = [ D2_x, ones(1,5000)'];
count = [0, 0, 0, 0, 0];
for k = 1:5
    w = ada_grad(D1_x,D1_y,learning_rate(k));
    for i = 1:5000
        temp = sign(dot(w,D2_x(i,:)));
        if temp == D2_y(i)
            count(k) = count(k) + 1;
        end
    end
end
% Training on 100% noisy data
[temp1,learning_ratei] = max(count);
learning_rate = learning_rate(learning_ratei);
T_x = [ T_x, ones(1,10000)'];
display(learning_rate)
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
