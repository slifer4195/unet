
close all;
clear;

%this file is to create new images and labels in certain size and change in

%resizing labels edge detection ///////////////////////////////////////////
OutputFolder = 'D:\SungRung\mnist_SEG(Noise)\project\data\resizedMnist';  % Set as needed [EDITED]
dinfo = dir('D:\SungRung\mnist_SEG(Noise)\project\data\oringalMnist\*.png');% image extension

for K = 1 :length(dinfo)-6000
    thisimage = dinfo(K).name;
    cd 'D:\SungRung\mnist_SEG(Noise)\project\data\oringalMnist'
    Img   = imread(thisimage);
    cd ..
    i = imresize(Img, [720, 960], 'bilinear');
    i = im2bw(i, 0.34);
    imwrite(i, fullfile(OutputFolder, thisimage));  % [EDITED]
end


%////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
OutputFolder = 'D:\SungRung\mnist_SEG(Noise)\project\data\noiseMnist';  % Set as needed [EDITED]
dinfo = dir('D:\SungRung\mnist_SEG(Noise)\project\data\resizedMnist\*.png');

for K = 1:length(dinfo)-6000
    thisimage = dinfo(K).name;
    cd 'D:\SungRung\mnist_SEG(Noise)\project\data\resizedMnist';
    input   = imread(thisimage);
    cd ..
    i1 = double(input);
    i1 = imnoise(i1,'speckle',20);
    i2 = imnoise(i1, 'salt & pepper', 0.45);
    i3 = imnoise(i2, 'gaussian', 0.45);
    imwrite(i3, fullfile(OutputFolder, thisimage))  % [EDITED]

end


