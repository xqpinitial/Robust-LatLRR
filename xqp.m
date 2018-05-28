
clc;
clear;
addpath inexact_alm_rpca;
addpath Ncut_9;
addpath inexact_alm_rpca/PROPACK;

video = VideoReader('Calib/data/morecars.avi');
A_hat_path='Calib/kA_hat/';
E_hat_path='Calib/kE_hat/';
U_hat_path='Calib/kU_hat/';
mkdir(U_hat_path)
Nframe = video.NumberOfFrames;  
H = video.Height/2;    
W = video.Width/2;     
Rate = video.FrameRate;
%read one frame every time
channel=3;
data=zeros(H*W*channel,Nframe);
for i = 1:Nframe
    Frame= read(video,i);
    Frame=im2double(imresize(Frame,[H,W]));
    imwrite(Frame,[U_hat_path,num2str(i),'.jpg']);
    data(:, i) = Frame(:); 
end
[m,n,channel]=size(Frame);
%%-------------------Data Generation-----------------------
lambda=1/sqrt(n*m*channel);
%% Robust LatLRR and RSIle
[U_hat Sigma_hat V_robust A_hat E_hat iter]= inexact_alm_rpca(data, lambda,1e-2,10); % The parameter lambda can be tuned.
[Z, ~] = RobustSelection(V_robust);
V = V_robust;
%%
A_hat=mat2gray(A_hat);
E_hat=mat2gray(E_hat);
mkdir(A_hat_path)
mkdir(E_hat_path)

for i = 1:Nframe
    ima=reshape(A_hat(:,i),[m,n,channel]);
    ime=reshape(E_hat(:,i),[m,n,channel]);
    imwrite(ima,[A_hat_path,num2str(i),'.jpg']);
    imwrite(ime,[E_hat_path,num2str(i),'.jpg'])

end

ima=reshape(A_hat(:,2),[m,n,channel]);
ime=reshape(E_hat(:,2),[m,n,channel]);
figure('name','A_hat'),imshow(ima)
figure('name','E_hat'),imshow(ime)
