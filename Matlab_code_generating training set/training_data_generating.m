clc;clear;close all;
% This m. file is used for generating virtual mixtures (1D&2D) as training set and validation set

%% load the data of pure substances and preprocess
% ---------- load data ---------- 
pure_mat = load('pure_ori_database_mat.mat');      % the spectra data of 100 pure substances saved in mat file ( a matrix, pure_num*spectral_wavenumber(100*1600 in paper)) , users should generate their own pure database.
pure_mat = pure_mat.pure;                          % 'pure' is the variable name of the matrix above 
pure_mat = pure_mat';
pure_name = load('pure_ori_database_name.mat');     % the name of 100 pure substances
pure_name = pure_name.pure_name;
pure_num = size(pure_mat,2);

% ---------- maximum normalization ---------- 
for l = 1:pure_num                              
    pure_mat(:,l) = pure_mat(:,l) / max(pure_mat(:,l));
end

%% generate the spectra of virtual mixtures (1D) and corresponding wavelet data (2D)
for i = 3      % or using loop like 1:6   
    fprintf('\n-----------------------------loop:%d----------------------------',i);
    vir_data = [];
    label_bc = [];
    index = 1;
    idx_exclude_cur = [[1:i-1],[i+1:pure_num]];    % the index of 99 substances in 100 except the selected specific substance
    cur_pure = pure_mat(:,i);                      % the index of the selected specific substance
    
    % ---------- positive virtual mixture ---------- 
    for j = 1:150                        % gen binary virtual mixture       
        rand_other_seq = randi(length(idx_exclude_cur),1);              
        cur_pure_other = pure_mat(:,idx_exclude_cur(rand_other_seq));    
        for k = 0.02:0.04:0.98                                          
            vir_data(:,index) = k*cur_pure + (1-k)*cur_pure_other;
            label_bc(index,:) = 1;
            index = index + 1;
        end
    end
    for jj = 1:250                       % gen tenary virtual mixture    
        rand_other_seq = randperm(length(idx_exclude_cur),2);
        for kk = 0.02:0.04:0.98
            ran_ratio = (1-kk)*rand();
            vir_data(:,index) = kk * cur_pure+ran_ratio*pure_mat(:,idx_exclude_cur(rand_other_seq(1)))+(1-kk-ran_ratio)*pure_mat(:,idx_exclude_cur(rand_other_seq(2)));
            label_bc(index,:) = 1;
            index = index + 1;
        end
    end
    
    % ---------- negative virtual mixture ---------- 
    for x = 1:150                         % gen binary virtual mixture
        rand_seq = randperm(length(idx_exclude_cur),2);     
        for xx = 0.02:0.04:0.98
            vir_data(:,index) = xx*pure_mat(:,idx_exclude_cur(rand_seq(1))) + (1-xx)*pure_mat(:,idx_exclude_cur(rand_seq(2)));
            label_bc(index,:) = 0;
            index = index + 1;
        end
    end    
    for x = 1:250                         % gen tenary virtual mixture
        random_ten = randperm(length(idx_exclude_cur),3);
        for xx = 0.02:0.04:0.98
            ran_ratio = (1-xx)*rand();
            vir_data(:,index) = xx*pure_mat(:,idx_exclude_cur(random_ten(1)))+ran_ratio*pure_mat(:,idx_exclude_cur(random_ten(2)))+(1-xx-ran_ratio)*pure_mat(:,idx_exclude_cur(random_ten(3)));
            label_bc(index,:) = 0;
            index = index + 1;
        end
    end
    
    rowrank = randperm(size(vir_data,2));     % shuffled
    vir_data = vir_data(:,rowrank);
    label_bc = label_bc';
    label_bc = label_bc(:,rowrank);
    
    for ii = 1 : size(vir_data,2)
        vir_data(:,ii) = vir_data(:,ii) / max(vir_data(:,ii));          % maximum normalization for virtual mixture
    end
    
    
    % transform the virtual spectra into wavelet data
    wav_vir_data = zeros( size(vir_data,2),50,length( vir_data(:,1) ) );
    for ii = 1 : size(vir_data,2)
        if mod(ii,1000) == 0
             fprintf('\n-------------------wav trans loop%d,%dth-------------------',i,ii);
        end
        wav_vir_data(ii,:,:) = cwt(vir_data(:,ii), (5:1:54), 'mexh');    % Notice! 'mexh' here means Lorentz4 wavelet, and ensure that "mexihat.m" is in the same path with this file when running!! 
    end    
    fprintf('\n');
    

    % set the save path and name
    vir_data_1D  = '../Matlab_data_training set/data';
    vir_data_Wav = '../Matlab_data_training set/wav_data';
    
    data_name = strcat(num2str(i),'component_data');
    label_name_bc = strcat(num2str(i),'component_label_bc');   
    
    % save the virtual data of 1D   
    if ~exist([vir_data_1D,num2str(i)],'dir')   
        mkdir([vir_data_1D,num2str(i)]);      
    end
    save([vir_data_1D,num2str(i),'\',data_name],'vir_data','-v7');
    save([vir_data_1D,num2str(i),'\',label_name_bc],'label_bc','-v7');
    
    % save the virtual data of wav
    if ~exist([vir_data_Wav,num2str(i)],'dir')
        mkdir([vir_data_Wav,num2str(i)]);
    end
    save([vir_data_Wav,num2str(i),'\',data_name],'wav_vir_data','-v7.3');
    save([vir_data_Wav,num2str(i),'\',label_name_bc],'label_bc','-v7.3');  
    
    clear vir_data    
    clear wav_vir_data
    clear label_bc

end