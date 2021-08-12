clear all;
close all;

dataFolder = "D:\SungRung\mnist_SEG(Noise)\project\data\testImages";
imds = imageDatastore(dataFolder,'IncludeSubfolders',true,'LabelSource','foldernames', 'ReadFcn', @to3D);
  
output = "D:\SungRung\mnist_SEG(Noise)\project\data\result\frameGeneratedImages";

data1 = load("Unet2.mat");
net1 = data1.net; 
% 
% data2 = load("Unet2.mat");
% net2 = data2.net; 
% 
% data3 = load("Unet3.mat");
% net3 = data3.net; 
% 
% data4 = load("Unet4.mat");
% net4 = data4.net; 
% 
% data5 = load("Unet5.mat");
% net5 = data5.net; 
% 
% data6 = load("Unet6.mat");
% net6 = data6.net; 
% 
% data7 = load("Unet7.mat");
% net7 = data7.net; 
% 
% data8 = load("Unet8.mat");
% net8 = data8.net; 
% 
% data9 = load("Unet9.mat");
% net9 = data9.net; 
% 
% data10 = load("Unet10.mat");
% net10 = data10.net; 

I2 = readimage(imds, 3);
I2 = rgb2gray(I2);
I2= imresize(I2, [720, 960],'bilinear');

I2 = im2bw(I2, 0.6);

I2 = I2 * 255;
I1 = cast(I2, 'uint8');
imwrite(I1, fullfile(output,"orignal.png"));

I1 = imnoise(I1, 'salt & pepper',0.47);  
I1 = imnoise(I1, 'gaussian',0.47);  

imshow(I1);
imwrite(I1, fullfile(output,"Noised.png"));

C1 = semanticseg(I1, net1); 
C1= cast(C1, 'double')-1;
imwrite(C1, fullfile(output,"badOutput1.png"));

pause;

%////////////////////////////////////////////////////////////////////////////////////////
C2 = semanticseg(I1, net2); 
C2= cast(C2, 'double')-1;
imwrite(C2, fullfile(output,"badOutput2.png"));

C3 = semanticseg(I1, net3); 
C3= cast(C3, 'double')-1;
% subplot(1,4,4);
% imshow(C3);
% title("net3")
imwrite(C3, fullfile(output,"badOutput3.png"));

C4 = semanticseg(I1, net4); 
C4= cast(C4, 'double')-1;

imwrite(C4, fullfile(output,"badOutput4.png"));
C5 = semanticseg(I1, net5); 
C5= cast(C5, 'double')-1;
imwrite(C5, fullfile(output,"badOutput5.png"));


C6 = semanticseg(I1, net6); 
C6= cast(C6, 'double')-1;
imwrite(C6, fullfile(output,"badOutput6.png"));

C7 = semanticseg(I1, net7); 
C7= cast(C7, 'double')-1;
imwrite(C7, fullfile(output,"badOutput7.png"));

C8 = semanticseg(I1, net8); 
C8= cast(C8, 'double')-1;
imwrite(C8, fullfile(output,"badOutput8.png"));

C9 = semanticseg(I1, net9); 
C9= cast(C9, 'double')-1;
imwrite(C9, fullfile(output,"badOutput9.png"));

C10 = semanticseg(I1, net10); 
C10 = cast(C10, 'double')-1;
imwrite(C10, fullfile(output,"badOutput10.png"));

newImage=I2 * 0.0;

newImage = double(newImage);
% newImage= (newImage + C1); newImage = imadjust(newImage);

newImage= (newImage + C2)/2;
newImage= (newImage + C3)/2;
newImage= (newImage + C4)/2;
newImage= (newImage + C5)/2;
newImage= (newImage + C6)/2;
newImage= (newImage + C7)/2;
newImage= (newImage + C8)/2;
newImage= (newImage + C9)/2;
newImage= (newImage + C10)/2;
% newImage = im2bw(newImage, 0.7);
imshow(newImage);


% newImage = newImage - 1.5;
% newImage=imadjust(newImage);
% subplot(1,3,2);
% imshow(newImage);
folder = "D:\SungRung\mnist_SEG(Noise)\project\data\result\frameGeneratedImages";
name = fullfile(folder, 'netCombined.png');
imwrite(newImage,  name);  
% title("denoised image")

function img = to3D(file)
     img = imread(file);
     if (size(img, 3) == 1)
     img= repmat(img, [1 1 3]);
     end

end