% Frankfurt University of Applied Sciences - FRA-UAS
% Faculty of Computer Sciences and Engineering 
% Machine Learning Project Winter Semester 2020/21 by Prof.Dr.Andreas Pech
% NGUYEN, HOANG LONG 1067534 hoang.nguyen3@stu.fra-uas.de
%

%% Import data from XLS Excel file - should only run once to save time !
clear; close; clc
A = readtable('combinedataforA.xlsx');
B = readtable('combinedataforB.xlsx');
%% Plot all the data of both Objects
close; clc
t = (1:width(A)); % Time vector 1:3400 samples

figure('Name','All 315 readings for Object A in time domain')
for c = 1:315  
S = table2array(A(c,1:width(A))); % Extract data from the table
S = S./max(max(S),abs(min(S))); % normalizing
plot(t,S); axis([1 width(A) -1 1]); xlabel('Samples'); ylabel('Amplitude')
drawnow; pause(0.01)
end %Object A

figure('Name','All 200 readings for Object B in time domain')
for c = 1:200  
S = table2array(B(c,1:width(A))); 
S = S./max(max(S),abs(min(S))); 
plot(t,S); axis([1 width(A) -1 1]); xlabel('Samples'); ylabel('Amplitude')
drawnow; pause(0.01)
end %Object B
%% Applying Gabor transformation
% Reference Nathn Kitz - Time Frequencey Analysis & Gabor transforms
% Applied Mathematics - University of Washington
close all

c = randi([1 315]); % Pulling a random reading from data of Object A
S = table2array(A(c,1:width(A))); 
S = S./max(max(S),abs(min(S))); % normalizing
figure('Name',"Reading "+c+" of Object A")
plot(t,S); axis([1 width(A) -1 1]); xlabel('Samples'); ylabel('Amplitude')

L = 10; %Signal duration = 10s assumption
n = 3400; % Number of samples
k = (2*pi/L)*[0:n/2-1 -n/2:-1]; ks = fftshift(k); 
%St = fft(S);

Sgt_spec = [];
tslide = 0:20:3400; % Moving Gabor filter every 20 samples !
figure('Name',"Gabor transform of reading "+c)
for j = 1:length(tslide)
g = exp(-5*1e-5*(t-tslide(j)).^2); %Gabor filter function
Sg = g.*S;
Sgt = fft(Sg);
Sgt_spec=[Sgt_spec; abs(fftshift(Sgt))];
subplot(3,1,1), plot(t,S,'k',t,g,'r')
title('Signal(back) and the Gabor filter(red)')
% xlabel('time'); ylable('amplitude')
subplot(3,1,2), plot(t,Sg,'k')
subplot(3,1,3), stem(ks,abs(fftshift(Sgt))/max(abs(fftshift(Sgt)))); %normalizing
axis([-100 100 0 1])
drawnow
pause(0.01)
end

% for i = 1:171
%     for j = 1:3400
%         Sgt_spec(i,j)= log(Sgt_spec(i,j)+1e-10);% log scaling
%     end
% end
Sgt_spec=Sgt_spec';% transpose the Spectrogram for better representation
figure('Name',"Spectrogram of reading "+c+" of Object A")
pcolor(tslide,ks,Sgt_spec),shading interp
set(gca,'Ylim',[30 60])
colormap default