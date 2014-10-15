% Configuration 1
l = 10;
m = [100,500,1000]; 
n = 1000;
pcount = zeros(1,3);
mpcount = zeros(1,3);
for a = 1:3
    % Generate data
    [y,x] = unba_gen(l,m(a),n,50000,0.1);
    [T_y,T_x] = unba_gen(l,m(a),n,10000,0.1);
    n = 0.1;

    % Perceptron
    % Tuning Step
    % No Paramter to tune
    % So Directly Training using 100% noisy data
    [w, theta ] = perceptron(x,y); 
    % Evaluation - Reporting the Error Rate
    for i = 1:10000
        temp = sign(dot(w,T_x(i,:)) + theta);
        if temp ~= T_y(i)
            pcount(a) = pcount(a) + 1;
        end
    end

    % Modified Perceptron
    % Tuning Step
    % No Paramter to tune
    % So Directly Training using 100% noisy data
    [w, theta ] = mod_perceptron(x,y,n); 
    % Evaluation - Reporting the Error Rate
    for i = 1:10000
        temp = sign(dot(w,T_x(i,:)) + theta);
        if temp ~= T_y(i)
            mpcount(a) = mpcount(a) + 1;
        end
    end
end

display(pcount/10000)
display(mpcount/10000)