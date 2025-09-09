function[EVAL1] = evaluation_pre(sp,act)
for i = 1:length(sp)
    p = sp{i};
    a = act{i};
    tp =0; tn =0; fp =0; fn =0;
    for j = 1:length(p)
        if a(j) == 1 && p(j) == 1
            tp = tp+1;
        elseif a(j) == 0 && p(j) == 0
            tn = tn+1;
        elseif a(j) == 0 && p(j) == 1
            fp = fp+1;
        elseif a(j) == 1 && p(j) == 0
            fn = fn+1;
        end
    end
    Tp(i) = tp;
    Fp(i) = fp;
    Tn(i) = tn;
    Fn(i) = fn;
end
tp = sum(Tp);
fp = sum(Fp);
tn = sum(Tn);
fn = sum(Fn);
Dice = (2 * tp) / ((2 * tp) + fp + fn)
Jaccard = tp / (tp + fp + fn)
accuracy = (tp+tn)/(tp+tn+fp+fn);

MSE = MeanSquareError(a, p);
PSNR = PeakSignaltoNoiseRatio(a, p);
IOU = (tp / (tp + fp + fn)) * 100;
EVAL1 = [Dice Jaccard accuracy MSE PSNR IOU];
end

function MSE = MeanSquareError(origImg, distImg)
origImg = double(origImg);
distImg = double(distImg);
[M N] = size(origImg);
error = origImg - distImg;
MSE = sum(sum(error .* error)) / (M * N);
end

function PSNR = PeakSignaltoNoiseRatio(origImg, distImg)
origImg = double(origImg);
distImg = double(distImg);
[M N] = size(origImg);
error = origImg - distImg;
MSE = sum(sum(error .* error)) / (M * N);
if(MSE > 0)
    PSNR = 10*log(255*255/MSE) / log(10);
else
    PSNR = 99;
end
end