% Configuration 1
l = 10;
m = 20; 
n = 40;
% Generate Clean data of 50000 examples and D1 and D2
[y,x] = gen(l,m,n,50000,0);

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
[w, theta ] = perceptron(D1_x,D1_y); % No tuning required
count = 0;
for i = 1:5000
    temp = sign(dot(w,D2_x(i,:)) + theta);
    if temp == D2_y(i)
        count = count + 1;
    end
end
display(count/5000);

% Perceptron with margin
learning_rate = [1.5, 0.25, 0.03, 0.005, 0.001];
count = [0, 0, 0, 0, 0];
for k = 1:5
    [w,theta] = perceptron_margin(D1_x,D1_y,learning_rate(k));
    for i = 1:5000
        temp = sign(dot(w,D2_x(i,:)) + theta);
        if temp == D2_y(i)
            count(k) = count(k) + 1;
        end
    end
end
display(count/5000)

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
display(count/5000)

% Winnow with margin
alpha =[1.1, 1.01, 1.005, 1.0005, 1.0001];
margin_parameter = [2.0, 0.3, 0.04, 0.006, 0.001];
count = [];
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
display(count/5000);
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
display(count/5000);
