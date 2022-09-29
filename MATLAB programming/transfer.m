load Depth=4_Epoch=100.mat  % 已訓練完之U-Net，當訓練出更好的結果，需更換
test_image = imread('./image_train/01416.tif');  % 欲測試的圖檔，即input
C = semanticseg(test_image, net);
B = labeloverlay(test_image,C, ...
    'Colormap','autumn','Transparency',0.25);  % 預測腫瘤區的疊圖，即output
figure
imshow(B)