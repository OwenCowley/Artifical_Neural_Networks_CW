% Find weightings by training the slp function (using the train data)
% Task 1 
%% First part
clc;clear all;close all;
load('class.mat');
% seed=10;
% W = slp(seed, Xtrain, ttrain, 0.005, 0.001, 10000,true);
 
% Wtrain=W; %Define the trained weights
% Uncomment above to find trained weights (below)
 
%% Set the trained weights (Task 1)
Wtrain = [-9.5105       % bias
           1.9353       % w1
           3.7436]      % w2
       
        
%% Use Wtrain to classify the test data (Task 2)
p= (Wtrain(1)+Wtrain(2)*Xtest(:,1)+Wtrain(3)*Xtest(:,2))<0;
class0 = Xtest(p==1,:);
 
q= (Wtrain(1)+Wtrain(2)*Xtest(:,1)+Wtrain(3)*Xtest(:,2))>0;
class1 = Xtest(q==1,:);
 
%% Create scatter plot (Task 3a)
class0x=class0(:,1);class0y=class0(:,2); scatter(class0x,class0y,'filled','r');hold on;
class1x=class1(:,1);class1y=class1(:,2); scatter(class1x,class1y,'filled','b');
 
%% Add line (Task 3b) 
xlim = get(gca,'XLim'); 
plot(xlim,[-(Wtrain(1)+Wtrain(2)*xlim(1))/Wtrain(3),  ...
    -(Wtrain(1)+Wtrain(2)*xlim(2))/Wtrain(3)],'k','LineWidth',2);
title('Scatter plot of test data');xlabel('x_1');ylabel('x_2');
hold off; 
 
%% Plot ttest data so assess results (Showing imperfect classification 
class0t = Xtest(ttest==0,:);
class1t  = Xtest(ttest==1,:);
 
figure;
 
class0xt=class0t(:,1);class0y=class0t(:,2); scatter(class0xt,class0y,'filled','r');hold on;
class1xt=class1t(:,1);class1y=class1t(:,2); scatter(class1xt,class1y,'filled','b');
 
xlim = get(gca,'XLim'); 
plot(xlim,[-(Wtrain(1)+Wtrain(2)*xlim(1))/Wtrain(3),  ...
    -(Wtrain(1)+Wtrain(2)*xlim(2))/Wtrain(3)],'k','LineWidth',2);
title('Scatter plot of ttest data');xlabel('x_1');ylabel('x_2');
hold off; 
 
%% Construct a confusion matrix from the test data (threshold =0.5) (Task 4)
y=slpFwd(Wtrain, Xtest);  %Output of the SLP
 
 
TP=size((Xtest(y>=0.5 & ttest==1)),1);
TN=size((Xtest(y<0.5 & ttest==0)),1);
FP=size((Xtest(y>=0.5 & ttest==0)),1);
FN=size((Xtest(y<0.5 & ttest==1)),1);
P= TP+FN; N= FP+TN;
NPR=100*(TN/(FN+TN));
PPR=100*(TP/(TP+FP));
TPR=100*(TP/P); Sensitivity=TPR;
FPR=100*(FP/N);
Specificity=(100-FPR);
Accuracy=100*((TP+TN)/(P+N));
 
Heading1 = {'Predicted_Positives';'Predicted_Negatives'};
Actual_Positives = [TP;FN];
Actual_Negatives = [FP;TN];
Confusion_matrix = table(Actual_Positives,Actual_Negatives,'RowNames',Heading1)
 
Heading_2 = {'PPR';'NPR';'Accuracy'; 'Specificity'; 'Sensitivity'};
Percentage = [PPR;NPR; Accuracy; Specificity; Sensitivity];
Performance_Indicators = table(Percentage,'RowNames',Heading_2)
 
%% Plot an ROC curve 
 
for s=1:11
    n=(s-1)/10;
TP2(s)=size((Xtest(y>=n & ttest==1)),1);
FP2(s)=size((Xtest(y>=n & ttest==0)),1);
 
P2(s)=size((Xtest(ttest==1)),1);
N2(s)=size([Xtest(ttest==0)],1);
 
TPR2(s)=100*(TP2(s)/P2(s));
FPR2(s)=100*(FP2(s)/N2(s));
 
roc=[FPR2; TPR2];
end
 
% subplot(1,2,2);
figure
stairs(roc(1,:), roc(2,:), 'b')
hold on 
scatter(roc(1,:), roc(2,:), 'r', 'filled')
title ('ROC Curve')
xlabel ('False Positive Rate')
ylabel ('True Positive Rate')
