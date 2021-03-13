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

% input length 3, 5, 7, 14

treatDate = 21;
%case 3
inputLengthChoice = input('Choose Input Length (indays) (1.3, 2.5, 3.7, 4.14): ');

% choose patient ID
switch inputLengthChoice
    case 1
        i = treatDate - 1; 
        j = treatDate + 1; 
    case 2
        i = treatDate - 2; 
        j = treatDate + 2; 
    case 3
        i = treatDate - 3; 
        j = treatDate + 3; 
    case 4
        i = treatDate - 6; 
        j = treatDate + 7; 
end

X = [hemo neut plt]';
X0 = X(:,i:i+j); %X_i
X1 = X(:,i+1:i+j+1); %X_(i+1)
sigma_X = X0*X0';
A_hat = X1*(X0')*(sigma_X^(-1));
e = X1 - A_hat*X0;
e_avg = sum(e,2)/j; %sample mean
Q_hat = ((e-e_avg)*(e-e_avg)')/(j-1); %sample covariance
W = Q_hat*Q_hat';
[Q,R] = qr(X0');
R22 = R(1:3,:);
gamma = 0.99;
c = (max(svd(inv(R22')*A_hat*R22'))/gamma - 1)/min(svd(inv(R22')*W*inv(R22))); %#ok<*MINV>
A_tilde = X1*X0'*(sigma_X+c*W)^(-1);


outputLength = j+ 20;

esti = X(:,1:j);
for i = j+1:outputLength
    esti(:,i) = A_tilde*X(:,i-1);
end

EstError = zeros(4,3);
errorCal1 = zeros(outputLength,1);
denominator = 0;
% figure;
% plot(hemo(1:end));
% hold on
% plot(neut(1:end));
% hold on
% plot(plt(1:end));
% legend('hemo',', neut', 'plt ');

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
for j = 1:4
    for i = 1:(outputLength - 5*(4-j)) 
         errorCal1(i) = abs(hemo(i)- esti(1,i)') / (sum(hemo(1:(outputLength - 5*(4-j)) ))/(outputLength - 5*(4-j)));
    end
    EstError(j,1) = var(errorCal1);
end
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

for j = 1:4
    for i = 1:(outputLength - 5*(4-j)) 
         errorCal2(i) = abs(neut(i)- esti(2,i)') / (sum(neut(1:(outputLength - 5*(4-j)) ))/(outputLength - 5*(4-j)));
    end
    EstError(j,2) = var(errorCal2);
end

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
for j = 1:4
    for i = 1:(outputLength - 5*(4-j)) 
         errorCal3(i) = abs(plt(i)- esti(3,i)') / (sum(plt(1:(outputLength - 5*(4-j)) ))/(outputLength - 5*(4-j)));
    end
    EstError(j,3) = var(errorCal3);
end
EstError(3) = var(errorCal3);


 

 