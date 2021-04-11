clc
clear
close all

load net.mat

imds = imageDatastore('spectrogram.jpg');
img = readimage(imds,1);
imageSize = size(img);
prediction = classify(net,imds)