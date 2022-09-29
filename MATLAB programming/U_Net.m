image_size = [256, 256, 3];
num_class = 2;
name_class = ["Tumor", "None"];
label = [255 0];

% Loading preprocessed data here
ds_image_train = imageDatastore('./image_train/');
ds_label_train = pixelLabelDatastore('./label_train/', name_class, label);
ds_image_valid = imageDatastore('./image_valid/');
ds_label_valid = pixelLabelDatastore('./label_valid/', name_class, label);
ds_image_test = imageDatastore('./image_test/');
ds_label_test = pixelLabelDatastore('./label_test/', name_class, label);
ds_train = combine(ds_image_train, ds_label_train);
ds_valid = combine(ds_image_valid, ds_label_valid);

% Building up and training model here
encoder_depth = 4;  % Modifying encoder depth here
epochs = 1;
lgraph = unetLayers(image_size, num_class, 'EncoderDepth', encoder_depth);
plot(lgraph)
options = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',epochs, ...
    'MiniBatchSize', 32, ...
    'ValidationData', ds_valid, ...
    'VerboseFrequency',5, ...
    'Plots', 'training-progress');  %'ExecutionEnvironment', 'multi-cpu'
[net,info] = trainNetwork(ds_train, lgraph, options);
save(strcat("Depth=",num2str(encoder_depth),"_Epoch=",num2str(epochs),".mat"),'net');  % Packaging parameters of trained model
ds_label_predict = semanticseg(ds_image_test, net, 'MiniBatchSize', 32, 'WriteLocation', tempdir);
metrics = evaluateSemanticSegmentation(ds_label_predict, ds_label_test);  % Evaluation of trained model



