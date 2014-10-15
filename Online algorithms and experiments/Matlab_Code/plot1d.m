% Set Configuration
l = 10;
m = 100; 
n = 1000;

% Generate Clean data of 50000 examples and D1 and D2
[y,x] = gen(l,m,n,50000,0);

% Perceptron
pcount =  zeros(1, 50000);
sumt = 0;
[w, theta ] = perceptron_1c(x,y); % No tuning required
for i = 1:50000
    temp = sign(dot(w,x(i,:)) + theta);
    if temp ~= y(i)
        sumt = sumt + 1;
    end
    pcount(i) = sumt;
end


% Perceptron with margin
learning_rate = 0.005;
pmcount =  zeros(1, 50000);
sumt = 0;
[w,theta] = perceptron_margin_1c(x,y,learning_rate);
for i = 1:50000
    temp = sign(dot(w,x(i,:)) + theta);
    if temp ~= y(i)
        sumt = sumt + 1;
    end
    pmcount(i) = sumt;    
end

% Winnow
alpha = 1.1;
wcount =  zeros(1, 50000);
sumt =0;
[w,theta] = winnow_1c(x,y,alpha);
for i = 1:50000
    temp = sign(dot(w,x(i,:)) + theta);
    if temp ~= y(i)
        sumt = sumt + 1;
    end
    wcount(i) = sumt;
end

% Winnow with margin
alpha =1.1;
margin_parameter = 2.0;
wmcount =  zeros(1, 50000);
sumt =0;
[w,theta] = winnow_margin_1c(x,y,alpha,margin_parameter);
for i = 1:50000
    temp = sign(dot(w,x(i,:)) + theta);
    if temp ~= y(i)
        sumt = sumt + 1;
    end
    wmcount(i) = sumt;
end

% AdaGrad
learning_rate = 0.25;
acount =  zeros(1, 50000);
sumt = 0;
w = ada_grad_1c(x,y,learning_rate);
x = [ x, ones(1,50000)'];
for i = 1:50000
    temp = sign(dot(w,x(i,:)));
    if temp ~= y(i)
        sumt = sumt + 1;
    end
    acount(i) = sumt;
end

% Plot the graph
figure;
hold on;
grid on;
title('Graph for 1c');
xlabel('Number of examples');
ylabel('Number of mistakes');   
plot(pcount,'--r');
plot(pmcount,'--g');
plot(wcount,'--b');
plot(wmcount,'--m');
plot(acount,'--k');
legend('Peceptron', 'Perceptron with Margin', 'Winnow', 'Winnow with Margin', 'AdaGrad')