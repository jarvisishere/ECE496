clear all;
patient = input('Choose Patient Data (1.SB, 2.MD, 3.TKB): ');

% choose patient ID
switch patient
    case 1
      disp('Patient ID:SB data loaded');
      hemo = load('SB_Hb.mat').SBHb;
      neut = load('SB_Neut.mat').SBNeut;
      plt = load('SB_Pit.mat').SBPit;
    case 2
       disp('Patient ID:MD data loaded');
       hemo = load('MD_Hb.mat').MDHb;
       neut = load('MD_Neut.mat').MDNeut;
       plt = load('MD_Pit.mat').MDPit;
    case 3
        disp('Patient ID:TKB data loaded')
        hemo = load('TKB_Hb.mat').TKBHb;
        neut = load('TKB_Neut.mat').TKBNeut;
        plt = load('TKB_Pit.mat').TKBPit;

end

treatment = load('SB_treat.mat').SBTreat;

% testingMode = input('Choose testing mode (1.Functinality, 2.Input date): ');
% switch testingMode
%     case 1
        i = 18; %time point at which the training starts
        j = i+7; % # of data poitns used for training
% end
x = [hemo neut plt]';
u = treatment';
len = length(x);
X = [x(:,1:len-2);x(:,2:len-1);x(:,3:len)];
U = [u(:,1:len-2);u(:,2:len-1);u(:,3:len)];

X0 = X(:,i:i+j);
X1 = X(:,i+1:i+j+1);
U0 = U(:,i:i+j);

sigma_XU = [X0*X0' X0*U0';U0*X0' U0*U0'];
AB_hat = X1*([X0' U0'])*(sigma_XU^(-1));
A_hat = AB_hat(:,1:9);
B_hat = AB_hat(:,10:15);

e = X1 - (A_hat*X0 + B_hat*U0);
e_avg = sum(e,2)/j; %sample mean
Q_hat = ((e-e_avg)*(e-e_avg)')/(j-1); %sample covariance
W = Q_hat*Q_hat';
We = [W zeros(9,6);zeros(6,9) zeros(6,6)];

[Q,R] = qr([U0' X0']);
R22 = R(7:15,7:15);

gamma = 0.99;

c = (max(svd(inv(R22')*A_hat*R22'))/gamma - 1)/min(svd(inv(R22')*W*inv(R22))); %#ok<*MINV>

AB_tilde = [X1*X0' X1*U0']*(sigma_XU+c*We)^(-1);
A_tilde = AB_tilde(:,1:9);
B_tilde = AB_tilde(:,10:15);
outputLength = j+40; % predict next 40 days
esti = X(:,1:j);
for i = j+1:outputLength
    esti(:,i) = A_tilde*X(:,i-1)+B_tilde*U(:,i-1);
end

EstError = [0, 0, 0 ];
errorCal1 = zeros(outputLength,1);
denominator = 0;
figure;
plot(hemo(1:end));
hold on
plot(neut(1:end));
hold on
plot(plt(1:end));
legend('hemo',', neut', 'plt ');

%plot data
figure;
subplot(3,2,1)
plot(esti(1,1:outputLength))
hold on
plot(hemo(1:outputLength))
switch patient
    case 1
        title("Patient ID: SB Hemoglobin Data")
        xline(21,'r--')
        xline(54,'r--')
    case 2
        title("Patient ID: MD Hemoglobin Data")
        xline(17,'r--')
        xline(55,'r--')
    case 3
        title("Patient ID: TKB Hemoglobin Data")
        xline(20,'r--')
        xline(70,'r--')
end
ylabel('Hemoglobin (g/L)'); 
xlabel('days');
legend('estimated data','actual data')
subplot(3,2,2)
% calculate estimation error
diff = hemo(1:outputLength)- esti(1,:)';
plot(abs(diff))
title("Hemoglobin Estimation Error")
ylabel('Hemo Difference (g/L)'); 
xlabel('days');
legend('estimated error');
for i = 1:outputLength 
    errorCal1(i) = abs(hemo(i)- esti(1,i)') / (sum(hemo(1:outputLength))/outputLength);
end
EstError(1) = var(errorCal1);
errorCal2 = zeros(outputLength,1);

subplot(3,2,3);
plot(esti(2,1:outputLength))
hold on
plot(neut(1:outputLength))
switch patient
    case 1
        title("Patient ID: SB Neutrophils Data")
        xline(21,'r--')
        xline(54,'r--')
    case 2
        title("Patient ID: MD Neutrophils Data")
        xline(17,'r--')
        xline(55,'r--')
    case 3
        title("Patient ID: TKB Neutrophils Data")
        xline(20,'r--')
        xline(70,'r--')
end
ylabel('Neutrophils *10^9(cells/L)'); 
xlabel('days');
legend('estimated data','actual data')
subplot(3,2,4)
% calculate estimation error
diff = neut(1:outputLength)- esti(2,:)';
plot(abs(diff))
title("Neutrophils Estimation Error")
ylabel('Neutrophils *10^9(cells/L)'); 
xlabel('days');
legend('estimated error');
for i = 1:outputLength 
    
    errorCal2(i) = abs(neut(i)- esti(2,i)') / (sum(neut(1:outputLength)) / outputLength);
   
end
 EstError(2) = var(errorCal2);
 errorCal3 = zeros(outputLength,1);

subplot(3,2,5);
plot(esti(3,1:outputLength))
hold on
plot(plt(1:outputLength))
switch patient
    case 1
        title("Patient ID: SB Platelets Data")
        xline(21,'r--')
        xline(54,'r--')
    case 2
        title("Patient ID: MD Platelets Data")
        xline(17,'r--')
        xline(55,'r--')
    case 3
        title("Patient ID: TKB Platelets Data")
        xline(20,'r--')
        xline(70,'r--')
end
ylabel('Platelets *10^9(cells/L)');
xlabel('days');
legend('estimated data','actual data')
% calculate estimation error
subplot(3,2,6)
diff = plt(1:outputLength)- esti(3,:)';
plot(abs(diff))
title("Platelets Estimation Error")
ylabel('Platelets *10^9(cells/L)'); 
xlabel('days');
legend('estimated error');
for i = 1:outputLength 
      errorCal3(i) = abs(plt(i)- esti(3,i)') /(sum(plt(1:outputLength))/outputLength);
end
EstError(3) = var(errorCal3);


 



