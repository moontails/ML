% Configuration 1
l = 10;
m = 20; 
n = [40, 80, 120, 160 200];

pm_learning = [0.25,0.25,0.03,0.05,0.03];
w_alpha = 1.1;
wm_alpha = 1.1;
wm_mp = 2.0;
ag_learning = 1.5;

pcount = zeros (1,5);
pmcount = zeros (1,5);
wcount = zeros (1,5);
wmcount = zeros (1,5);
acount = zeros (1,5);
% Perceptron
for i = 1:5
    [y,x] = gen(l,m,n(i),50000,0);
    pcount(i) = perceptron_2(x,y); % No tuning required
    pmcount(i) = perceptron_margin_2(x,y,pm_learning(i));
    wcount(i) = winnow_2(x,y,w_alpha);
    wmcount(i) = winnow_margin_2(x,y,wm_alpha,wm_mp);
    acount(i) = ada_grad_2(x,y,ag_learning);
end
% Plot the graph
figure;
hold on;
grid on;
title('Graph for 2');
xlabel('Number of examples');
ylabel('Number of mistakes');   
set(gca,'Xtick',40:40:200);
plot(n,pcount,'--r');
plot(n,pmcount,'--g');
plot(n,wcount,'--b');
plot(n,wmcount,'--m');
plot(n,acount,'--k');
legend('Peceptron', 'Perceptron with Margin', 'Winnow', 'Winnow with Margin', 'AdaGrad')
