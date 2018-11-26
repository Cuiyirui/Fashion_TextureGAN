%%
clear;clc;close all;
noise_thre=230;
%%
I = imread('../datasets/contour2shirt/train/10.jpg');
contour = I(:,1:256,:);
contour = rgb2gray(contour);

%refine contour
BW2=contour<noise_thre;
figure,imshow(BW2);
BW3 = imfill(BW2,'holes');
figure,imshow(BW3)

