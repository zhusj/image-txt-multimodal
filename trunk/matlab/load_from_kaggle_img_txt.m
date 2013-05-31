%{
Copyright (C) 2013 Yichuan Tang. 
contact: tang at cs.toronto.edu
http://www.cs.toronto.edu/~tang

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
%}

clear;
%% first load all the words from the ESP

FILEPATH = '/ais/gobi3/u/tang/ESPGame100k/labels/';
sDir = dir(FILEPATH);

if 1
    dict = containers.Map;
    for ss = 3:length(sDir)

        gprintf('%d', ss);
        fid = fopen([FILEPATH sDir(ss).name]);
        dd = textscan(fid, '%s');
        dd = dd{1};
        fclose(fid);
        for d=1:length(dd)
            if dict.isKey(dd{d})
                dict(dd{d}) = dict(dd{d})+1;
            else
                dict(dd{d}) = 1;
            end
        end
    end
    save esp_dictionary.mat dict;
end

if 1
    load esp_dictionary.mat;
    vv = values(dict);    
    vv2 = cell2mat(vv);    
    [val ind] = sort(vv2,'descend');
    
    freq=[];
    words = dict.keys();
    for i = 1:1024
        freq{i} = words{ind(i)};        
    end
    
    
    %% get labels for training        
    DataY = zeros(100000, 1024, 'single');
    DataX = zeros(100000, 48^2*3, 'single');
    
    IMAGEPATH = '/ais/gobi3/u/tang/ESPGame100k/originals/';
    dict2 =containers.Map(freq, 1:1024);
    
    bad =[];
    
    for ss = 3:length(sDir)
        gprintf('%d', ss);
        fid = fopen([FILEPATH sDir(ss).name]);
        dd = textscan(fid, '%s');
        dd = dd{1};
        fclose(fid);
        for d =1:length(dd)            
           if dict2.isKey(dd{d})           
               DataY(ss-2, dict2(dd{d}) ) = 1;
           end
        end
        
        try
            if exist([IMAGEPATH sDir(ss).name(1:end-5)], 'file')            
                im = imread([IMAGEPATH sDir(ss).name(1:end-5)]);
            else
                im = imread([IMAGEPATH sDir(ss).name(1:end-8) 'gif']);
                if length(size(im))==4
                    im = im(:,:,:,1);
                end         
            end        
        catch err
            bad = [bad ss-2]
            continue;
        end
        
        im = imresize(single(im)./255, [48,48]);
        if length(size(im))==2
            im = repmat(im,[1 1 3]);
        end
        DataX(ss-2,:) = sc(permute(im,[2 1 3]));
    end        
    save esp_data.mat freq dict2 bad DataX DataY -v7.3;    
end


%% test images

sDir = dir('public_test_images');

TestX = zeros(500, 48^2*3, 'single');
TestX0 = zeros(500, 1024, 'single');
TestX1 = zeros(500, 1024, 'single');

%TestInd =zeros(500,1);

for ss = 3:length(sDir) 
    
    gprintf('%d', ss);
    
    %TestInd(ss-2) = str2num(sDir(ss).name(1:end-4));
    realind = int32(str2num(sDir(ss).name(1:end-4)))+1;
      
    im = imread(['public_test_images/' sDir(ss).name]);    
    im = imresize(single(im)./255, [48,48]);
    if length(size(im))==2
        im = repmat(im,[1 1 3]);
    end
    TestX(realind,:) = sc(permute(im,[2 1 3]));
    
    fid = fopen(['public_test_options/' sDir(ss).name(1:end-3) 'option_0.desc']);
    dd = textscan(fid, '%s');
    dd = dd{1};
    fclose(fid);
    for d =1:length(dd)
        if dict2.isKey(dd{d})
            TestX0(realind, dict2(dd{d}) ) = 1;
        end
    end    
    
    fid = fopen(['public_test_options/' sDir(ss).name(1:end-3) 'option_1.desc']);
    dd = textscan(fid, '%s');
    dd = dd{1};
    fclose(fid);
    
    for d =1:length(dd)
        if dict2.isKey(dd{d})
            TestX1(realind, dict2(dd{d}) ) = 1;
        end
    end    
end

save test.mat TestX TestX0 TestX1 -v7.3;

%%
sDir = dir('private_test_images');

TestX = zeros(500, 48^2*3, 'single');
TestX0 = zeros(500, 1024, 'single');
TestX1 = zeros(500, 1024, 'single');

%TestInd =zeros(500,1);

for ss = 3:length(sDir) 
    
    gprintf('%d', ss);
    
    %TestInd(ss-2) = str2num(sDir(ss).name(1:end-4));
    realind = int32(str2num(sDir(ss).name(1:end-4)))+1;
      
    im = imread(['private_test_images/' sDir(ss).name]);    
    im = imresize(single(im)./255, [48,48]);
    if length(size(im))==2
        im = repmat(im,[1 1 3]);
    end
    TestX(realind,:) = sc(permute(im,[2 1 3]));
    
    fid = fopen(['private_test_options/' sDir(ss).name(1:end-3) 'option_0.desc']);
    dd = textscan(fid, '%s');
    dd = dd{1};
    fclose(fid);
    for d =1:length(dd)
        if dict2.isKey(dd{d})
            TestX0(realind, dict2(dd{d}) ) = 1;
        end
    end    
    
    fid = fopen(['private_test_options/' sDir(ss).name(1:end-3) 'option_1.desc']);
    dd = textscan(fid, '%s');
    dd = dd{1};
    fclose(fid);
    
    for d =1:length(dd)
        if dict2.isKey(dd{d})
            TestX1(realind, dict2(dd{d}) ) = 1;
        end
    end
    
end

save privatetest.mat TestX TestX0 TestX1 -v7.3;


