%% Image HW04 - Ali Ghavampour - 97102293

%% Question 1 - Part 1 - Images
clear all; close all; clc;
img1 = im2double(imread('Mri1.bmp'));
img2 = im2double(imread('Mri2.bmp'));
img3 = im2double(imread('Mri3.bmp'));
img4 = im2double(imread('Mri4.bmp'));
img5 = im2double(imread('Mri5.bmp'));

rgb1(:,:,1) = img1;
rgb1(:,:,2) = img2;
rgb1(:,:,3) = img3;

rgb2(:,:,1) = img3;
rgb2(:,:,2) = img4;
rgb2(:,:,3) = img5;

rgb3(:,:,1) = img1;
rgb3(:,:,2) = img2;
rgb3(:,:,3) = img4;

rgb4(:,:,1) = img3;
rgb4(:,:,2) = img2;
rgb4(:,:,3) = img5;

montage({rgb1,rgb2,rgb3,rgb4})

%% Question 1 - Part 2 - FCM
clear all; close all; clc;
img = im2double(imread('Mri1.bmp'));
sz = size(img);
vec = reshape(img,sz(1)*sz(2),1);

options = [1.4 NaN NaN NaN];
[centers,U] = fcm(vec,4,options);
u1 = reshape(U(1,:),256,256);
u2 = reshape(U(2,:),256,256);
u3 = reshape(U(3,:),256,256);
u4 = reshape(U(4,:),256,256);
montage({u1,u2,u3,u4},'size',[1,4])
title("p = 1.4")

color1 = zeros(sz(1),sz(2),3);
color2 = zeros(sz(1),sz(2),3);
color3 = zeros(sz(1),sz(2),3);
color4 = zeros(sz(1),sz(2),3);
for i = 1:sz(1)
    for j = 1:sz(2)
        color1(i,j,:) = u1(i,j) * [1,0,0];
        color2(i,j,:) = u2(i,j) * [0,1,0];
        color3(i,j,:) = u3(i,j) * [0,0,1];
        color4(i,j,:) = u4(i,j) * [1,1,1];
    end
end
colorU1 = color1+color2+color3+color4;


options = [5 NaN NaN NaN];
[centers,U] = fcm(vec,4,options);
u1 = reshape(U(1,:),256,256);
u2 = reshape(U(2,:),256,256);
u3 = reshape(U(3,:),256,256);
u4 = reshape(U(4,:),256,256);
figure;
montage({u1,u2,u3,u4},'size',[1,4])
title("p = 5")

color1 = zeros(sz(1),sz(2),3);
color2 = zeros(sz(1),sz(2),3);
color3 = zeros(sz(1),sz(2),3);
color4 = zeros(sz(1),sz(2),3);
for i = 1:sz(1)
    for j = 1:sz(2)
        color1(i,j,:) = u1(i,j) * [1,0,0];
        color2(i,j,:) = u2(i,j) * [0,1,0];
        color3(i,j,:) = u3(i,j) * [0,0,1];
        color4(i,j,:) = u4(i,j) * [1,1,1];
    end
end
colorU2 = color1+color2+color3+color4;

options = [1.1 NaN NaN NaN];
[centers,U] = fcm(vec,4,options);
u1 = reshape(U(1,:),256,256);
u2 = reshape(U(2,:),256,256);
u3 = reshape(U(3,:),256,256);
u4 = reshape(U(4,:),256,256);
figure;
montage({u1,u2,u3,u4},'size',[1,4])
title("p = 1.1")

color1 = zeros(sz(1),sz(2),3);
color2 = zeros(sz(1),sz(2),3);
color3 = zeros(sz(1),sz(2),3);
color4 = zeros(sz(1),sz(2),3);
for i = 1:sz(1)
    for j = 1:sz(2)
        color1(i,j,:) = u1(i,j) * [1,0,0];
        color2(i,j,:) = u4(i,j) * [0,1,0];
        color3(i,j,:) = u3(i,j) * [0,0,1];
        color4(i,j,:) = u2(i,j) * [1,1,1];
    end
end
colorU3 = color1+color2+color3+color4;

figure;
montage({colorU3,colorU1,colorU2},'size',[1,3]);
title("Segmented Image - Left: p=1.1 , Middle: p=1.4 , Right: p=5")

%% Question 1 - Part 3 - kmeans initial condition
clear all; close all; clc;
img = im2double(imread('Mri1.bmp'));
sz = size(img);
vec = reshape(img,sz(1)*sz(2),1);

tic
[idx,C] = kmeans(vec,4);
t_kmeans = toc;
cluster = reshape(idx,sz(1),sz(2));
c = {[1,0,0],[0,1,0],[0,0,1],[1,1,1]};
tmp = zeros(sz(1),sz(2),3);
for i = 1:sz(1)
    for j = 1:sz(2)
        ind = cluster(i,j);
        color = c{ind};
        tmp(i,j,:) = color;
    end
end
imshow(tmp)
title("K-Means result")

% P = 1.4
options = [1.4 NaN NaN NaN];
tic
[centers,U] = myfcm(vec,4,options,idx);
t_kmeans = t_kmeans + toc;
u1 = reshape(U(1,:),256,256);
u2 = reshape(U(2,:),256,256);
u3 = reshape(U(3,:),256,256);
u4 = reshape(U(4,:),256,256);
colorU141 = fcmSegmenter(u1,u2,u3,u4);

options = [1.4 NaN NaN NaN];
tic
[centers,U] = fcm(vec,4,options);
t_fcm = toc;
u1 = reshape(U(1,:),256,256);
u2 = reshape(U(2,:),256,256);
u3 = reshape(U(3,:),256,256);
u4 = reshape(U(4,:),256,256);
colorU142 = fcmSegmenter(u1,u2,u3,u4);
fprintf("P = 1.4: \n")
fprintf("time with kmeans initial values = %.4f\n",t_kmeans)
fprintf("time of only fcm = %.4f\n",t_fcm)


% P = 1.1
options = [1.1 NaN NaN NaN];
tic
[centers,U] = myfcm(vec,4,options,idx);
t_kmeans = t_kmeans + toc;
u1 = reshape(U(1,:),256,256);
u2 = reshape(U(2,:),256,256);
u3 = reshape(U(3,:),256,256);
u4 = reshape(U(4,:),256,256);
colorU111 = fcmSegmenter(u1,u2,u3,u4);

options = [1.1 NaN NaN NaN];
tic
[centers,U] = fcm(vec,4,options);
t_fcm = toc;
u1 = reshape(U(1,:),256,256);
u2 = reshape(U(2,:),256,256);
u3 = reshape(U(3,:),256,256);
u4 = reshape(U(4,:),256,256);
colorU112 = fcmSegmenter(u1,u2,u3,u4);
fprintf("P = 1.1: \n")
fprintf("time with kmeans initial values = %.4f\n",t_kmeans)
fprintf("time of only fcm = %.4f\n",t_fcm)


% p = 5
options = [5 NaN NaN NaN];
tic
[centers,U] = myfcm(vec,4,options,idx);
t_kmeans = t_kmeans + toc;
u1 = reshape(U(1,:),256,256);
u2 = reshape(U(2,:),256,256);
u3 = reshape(U(3,:),256,256);
u4 = reshape(U(4,:),256,256);
colorU51 = fcmSegmenter(u1,u2,u3,u4);

options = [5 NaN NaN NaN];
tic
[centers,U] = fcm(vec,4,options);
t_fcm = toc;
u1 = reshape(U(1,:),256,256);
u2 = reshape(U(2,:),256,256);
u3 = reshape(U(3,:),256,256);
u4 = reshape(U(4,:),256,256);
colorU52 = fcmSegmenter(u1,u2,u3,u4);
fprintf("P = 5: \n")
fprintf("time with kmeans initial values = %.4f\n",t_kmeans)
fprintf("time of only fcm = %.4f\n",t_fcm)

figure;
montage({colorU111,colorU112,colorU141,colorU142,colorU51,colorU52},'size',[3,2])
title("Left: With K-Means Initilization - Right: With Random Initilization")
ylabel("Top: p=1.1 , Middle: p=1.4 , Bottom: p=5")


%% Question 1 - Part 4 - GMM fit
clear all; close all; clc;
rng shuffle
img = im2double(imread('Mri1.bmp'));
sz = size(img);
vec = reshape(img,sz(1)*sz(2),1);
options = [1.4 NaN NaN NaN];
[~,U] = fcm(vec,4,options);
fcmStart = zeros(sz(1)*sz(2),1);
for i = 1:sz(1)*sz(2)
    maxU = max(U(:,i));
    ind = find(U(:,i)==maxU);
    fcmStart(i) = ind;
end

nbins = 100;
[counts,centers] = hist(vec,nbins);
% counts(1) = 0;
counts = counts/max(counts);
bar(centers,counts,1)

GMModel = fitgmdist(vec,4,'start',fcmStart,'RegularizationValue',0.00001)
meanGM = GMModel.mu;
sigmaGM = GMModel.Sigma;
sigmaGM = sigmaGM(:);
x = linspace(0,1,nbins);
y = pdf(GMModel,x');
y = y/max(y);
hold on
plot(x,y,'r','linewidth',2)
title("GM Model fitted to the image histogram")

hold all
plot([meanGM(1),meanGM(1)],[0,0.2],'--r','linewidth',2)
plot([meanGM(2),meanGM(2)],[0,0.2],'--r','linewidth',2)
plot([meanGM(3),meanGM(3)],[0,0.2],'--r','linewidth',2)
plot([meanGM(4),meanGM(4)],[0,0.2],'--r','linewidth',2)


% segmentation
g1 = makedist('Normal','mu',meanGM(1),'sigma',0.1);
g2 = makedist('Normal','mu',meanGM(2),'sigma',0.1);
g3 = makedist('Normal','mu',meanGM(3),'sigma',0.1);
g4 = makedist('Normal','mu',meanGM(4),'sigma',0.1);
segment = zeros(sz(1)*sz(2),1);

p1 = cdf(g1,vec)-cdf(g1,vec-0.001);
p2 = cdf(g2,vec)-cdf(g2,vec-0.001);
p3 = cdf(g3,vec)-cdf(g3,vec-0.001);
p4 = cdf(g4,vec)-cdf(g4,vec-0.001);
for i = 1:sz(1)*sz(2)
    p = [p1(i),p2(i),p3(i),p4(i)];
    maxP = max(p);
    ind = find(p==maxP);
    segment(i) = ind;
end
segment = reshape(segment,sz(1),sz(2));

c = {[1,0,0],[0,1,0],[0,0,1],[1,1,1]};
tmp = zeros(sz(1),sz(2),3);
for i = 1:sz(1)
    for j = 1:sz(2)
        ind = segment(i,j);
        color = c{ind};
        tmp(i,j,:) = color;
    end
end
figure
imshow(tmp)
title("Segmentation Using GMM")


%% Question 2 =============================================================

%% GVF
clear all; close all; clc;
cd '.\snake_demo'
sdemo

%% Basic Snake
clear all; close all; clc;
cd '.\activeContoursSnakesDemo\activeContoursDemo'
snk

%% Question 3 ============================================================
clear all; close all; clc;
cd '.\ARKFCM_demo\ARKFCM_demo\ARKFCM_demo'
demo



%% functions
function colorU = fcmSegmenter(u1,u2,u3,u4)
    sz = size(u1);
    color1 = zeros(sz(1),sz(2),3);
    color2 = zeros(sz(1),sz(2),3);
    color3 = zeros(sz(1),sz(2),3);
    color4 = zeros(sz(1),sz(2),3);
    for i = 1:sz(1)
        for j = 1:sz(2)
            color1(i,j,:) = u1(i,j) * [1,0,0];
            color2(i,j,:) = u2(i,j) * [0,1,0];
            color3(i,j,:) = u3(i,j) * [0,0,1];
            color4(i,j,:) = u4(i,j) * [1,1,1];
        end
    end
    colorU = color1+color2+color3+color4;
end












