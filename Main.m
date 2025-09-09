function [] = Main()
clc;
clear all;
close all;
warning off
global Feat1 Feat2 Targets1 Targets2 Bestsol Train_Data Test_Data Train_Target Test_Target prep1 prep2

%% Read Dataset 1
an = 0;
if an == 1
    Directory = './Dataset/Dataset 1';
    Target1 = [];
    in_folder = dir(Directory);
    in_folder = in_folder([in_folder.isdir]); % Get only directories
    in_folder = in_folder(3:end); % Skip '.' and '..'
    iteration = 1;
    for j = 1:length(in_folder)
        indir_1 = fullfile(Directory, in_folder(j).name);
        in_fold = dir(indir_1);
        in_fold = in_fold(~[in_fold.isdir]); % Get only files
        for k = 1:length(in_fold)
            filename = fullfile(indir_1, in_fold(k).name);
            image = Read_Image(filename);
            image = imresize(image, [128 128]);
            Images1{iteration} = image;
            iteration = iteration + 1;
            Target1 = [Target1; j];
        end
    end
    tar = Target1;
    uniq = unique(tar);
    Targets1 = zeros(length(tar), length(uniq));
    for uni = 1:length(uniq)
        index = find(tar == uniq(uni));
        Targets1(index, uni) = 1;
    end
    
    save Images1 Images1
    save Targets1 Targets1
end

%% Read Dataset 2
an = 0;
if an == 1
    Directory_img = './Dataset/Dataset 2/gCWWgc8puP';
    Directory_txt = './Dataset/Dataset 2/Er1XzGfJkW.txt';
    data = importdata(Directory_txt) ;  
    % Get a list of all files in the folder with .tiff extension
    imageFiles = dir(fullfile(Directory_img, '*.tiff')); 
    Target2 = [];
    iteration = 1;
    % Loop through each image in the folder
    for k = 1:length(imageFiles)
        % Get the full filename of the image
        imageFile = fullfile(Directory_img, imageFiles(k).name);
        % Read the .tiff image
        img = Read_Image(imageFile);
        image = imresize(img, [128 128]);
        Images2{iteration} = image;
        img_name = imageFiles(k).name;
        iteration = iteration + 1;
%         isPresent = ismember(data.textdata, );
        isPresent = ismember(img_name(1:end-5), data.textdata);
       if isPresent
            position = find(strcmp(data.textdata, img_name(1:end-5)));
            Target2 = [Target2; data.data(position)];      
        end
    end
    tar = Target2;
    uniq = unique(tar);
    Targets2 = zeros(length(tar), length(uniq));
    for uni = 1:length(uniq)
        index = find(tar == uniq(uni));
        Targets2(index, uni) = 1;
    end
    save Images2 Images2
    save Targets2 Targets2
end

%% Preprocess Dataset 1
an = 0;
if an == 1
    load Images1
    load Images2
    iteration = 1;
    for i = 1:length(Images1)
        disp(i)
        A = Images1{i};
        Contrast = prep_pro(A);
        Images1{iteration} = Contrast;
        iteration = iteration + 1;
        prep1 = Images1;
    end
    iter = 1;
    for i = 1:length(Images2)
        disp(i)
        A = Images2{i};
        Contrast = prep_pro(A);
        Images2{iter} = Contrast;
        iter = iter + 1;
        prep2 = Images2;
    end
    save prep1 prep1 
    save prep2 prep2
end 
%% Segmentation For Transformer Recurrent Residual U-Net
an = 0;
if an == 1
    load prep1
    load prep2
    iteration = 1;
    for i = 1:length(prep1)
        disp(i)
        image1 = prep1;
        Segment = Model_TR2Unet(image1); % Transformer Recurrent Residual U-Net
        Images1{iteration} = Segment;
        iteration = iteration + 1;
        Seg_Image1 = Images1;
    end
    iter = 1;
    for i = 1:length(prep2)
        disp(i)
        image2 = prep2;
        Segment = Model_TR2Unet(image2); % Transformer Recurrent Residual U-Net
        Images2{iter} = Segment;
        iter = iter + 1;
        Seg_Image2 = Images2;
    end
    save Seg_Image1 Seg_Image1
    save Seg_Image2 Seg_Image2
end
%% Optimization for Classification
an = 1;
if an == 1
    load Seg_Image1
    load Seg_Image2
    load Targets1
    load Targets2
    for k = 1:2 
        Feat1 = Seg_Image1;
        Feat2 = Seg_Image2;
        tar2 = Targets1;
        tar2 = Targets2;

        Npop = 10; % Population size
        Ch_len = 3;  % No.of feature to be selected
        xmin = [5, 5, 50] .* ones(Npop, Ch_len);  % Minimum Bound
        xmax = [255, 50, 250] .* ones(Npop, Ch_len);   % maxiumum Bound
        initsol = unifrnd(xmin,xmax);
        itermax = 50;
        fname = 'Objfun_Cls';

        disp('CO');
        hist = zeros(1,size(initsol,2)); fit_hist = 0;
        [bestfit,fitness,bestsol,time] = PSO(initsol,fname,xmin,xmax,itermax);
        Co(k).bf = bestfit; Co(k).fit = fitness; Co(k).bs = bestsol; Co(k).ct = time;
        save Co Co

        disp('AOA');
        hist = zeros(1,size(initsol,2)); fit_hist = 0;
        [bestfit,fitness,bestsol,time] = PSO(initsol,fname,xmin,xmax,itermax);
        Aoa(k).bf = bestfit; Aoa(k).fit = fitness; Aoa(k).bs = bestsol; Aoa(k).ct = time;
        save Aoa Aoa

        disp('WSA');
        hist = zeros(1,size(initsol,2)); fit_hist = 0;
        [bestfit,fitness,bestsol,time] = PSO(initsol,fname,xmin,xmax,itermax);
        Wsa(k).bf = bestfit; Wsa(k).fit = fitness; Wsa(k).bs = bestsol; Wsa(k).ct = time;
        save Wsa Wsa

        disp('MOA');
        hist = zeros(1,size(initsol,2)); fit_hist = 0;
        [bestfit,fitness,bestsol,time] = PSO(initsol,fname,xmin,xmax,itermax);
        Moa(k).bf = bestfit; Moa(k).fit = fitness; Moa(k).bs = bestsol; Moa(k).ct = time;
        save Moa Moa

        disp('PROPOSED');
        hist = zeros(1,size(initsol,2)); fit_hist = 0;
        [bestfit,fitness,bestsol,time] = PSO(initsol,fname,xmin,xmax,itermax);  % PROPOSED
        Prop(k).bf = bestfit; Prop(k).fit = fitness; Prop(k).bs = bestsol; Prop(k).ct = time;
        save Prop Prop
    end
end

%% Classification
an = 0;
if an == 1
    disp('Classification')
    load Seg_Image1
    load Seg_Image2
    load Targets1
    load Targets2
    for k = 1:2
        if k == 1
            Targets = Targets1;
            Feat = Seg_Image1;
        else
            Targets = Targets2;
            Feat = Seg_Image2;
        end 
        Algo = {'Co', 'Aoa', 'Wsa', 'Moa', 'Prop'};
        learnper = [0.35 0.45 0.55 0.65 0.75];
        for i = 1:length(learnper)
            learnperc = round(length(Targets) * learnper(i));
            for j = 1:length(Algo)
                eval(['load ', char(Algo{i}), 'Net'])
                eval(['alg = ', char(Algo{i}), ';'])
                Bestsol = round(alg.bs);
                Train_Data = Feat(1:learnperc, :);
                Train_Target = Targets(1:learnperc, :);
                Test_Data = Feat(learnperc + 1:end, :);
                Test_Target = Targets(learnperc + 1:end, :);
                [Eval(j, :), Net_out{i, j}] = Model_Proposed();
            end
            [Eval(6, :), Net_out{i, 6}] = Model_Resnet();
            [Eval(7, :), Net_out{i, 7}] = Model_CNN();
            [Eval(8, :), Net_out{i, 8}] = Model_LSTM();
            sol = Bestsol;
            Bestsol = [5, 5, 50];
            [Eval(9, :), Net_out{i, 9}] = Model_Proposed();
            Bestsol = sol;
            [Eval(10, :), Net_out{i, 10}] = Model_Proposed();
            Ev{i} = Eval;
        end
        Eval_all{k} = Ev;
    end
   save Eval_all Eval_all
end
Plot_Results()
end

function Contrast = prep_pro(A)
patch = imcrop(A,[170, 150, 150 150]);
patchVar = std2(patch)^2;
DoS = 2*patchVar;
% DoS = 0.1;
J = imbilatfilt(A,DoS);
edgeThreshold =  0.8;
amount = 1;
Contrast = localcontrast(J, edgeThreshold, amount); % Contrast enchancement
end


function image = Read_Image(filename)
image = imread(filename);
if size(image, 3) == 3
    image = rgb2gray(image);
end
end
