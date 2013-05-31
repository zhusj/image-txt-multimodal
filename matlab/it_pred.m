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


rg;
clear all;

% parameters
TESTDATA_FILE = 'privatetest.mat';
MODEL_FILE = 'model.it.5.31.mat';

gg;
addpath('../cuda_ut/bin');
% load data
load(TESTDATA_FILE);
models = load(MODEL_FILE);

%% test

global D;
global Dy;

D =[];
Dy =[];
D{1} = models.hp.D;
Dy{1} = models.hp.Dy;

TestPredY = zeros(size(TestX,1), 1024, length(models.cv_models));

for mm = 1:length(models.cv_models)
    
    fprintf('\n split:%d ', mm);
    model = models.cv_models{mm};
    nSamples = models.hp.nSamples;

    nTest = size(TestX,1)*size(TestX,3);
    nbatches2 = ceil( nTest/nSamples);
    nrepmat2 = nbatches2*nSamples-nTest;
    nJit = size(TestX,3);
    
    global input;
    input =[];
    input.X{1} = [batchdata_reshape(TestX(:,:,:)); ...
                  zeros( nrepmat2, size(TestX,2))];
   
    input.Y{1} = zeros(size(input.X{1},1), Dy{1});
        
    model.nSamples = nSamples;
    model.nLayerEst = length(model.net_layers)-1;
    model.MODE = 'classify';
    model.nTestingMode = 1;
    
    [~, res] = myclassify_conv_nn_softmax(model);
    
    Yest2 = zeros(floor(nTest/nJit), Dy{1});
    
    for kk = 1:length(res.y_est)  
        Yest = double(batchdata_reshape(res.y_est{kk}));

        Yest = batchdata_reshape(Yest(1:nTest,:), ...
                [floor(nTest/nJit), Dy{1}, nJit]);

        for k = 1:nJit,
            temp = double(squeeze(Yest(:,:,k)));            
            Yest2 = Yest2 + temp;
        end
    end
    
    Yest2 = Yest2./(length(res.y_est)*nJit);          
    TestPredY(:,:,mm) =Yest2;   
end

TestPredY2 = mean(TestPredY,3);


%%
Yest2 = max( min(TestPredY2, 1-1e-8), 1e-8);

ce = double(TestX0).*log(Yest2)+ ...
    (1-TestX0).*log(1-Yest2);
ce0 = sum(ce,2);

ce = double(TestX1).*log(Yest2)+ ...
    (1-TestX1).*log(1-Yest2);
ce1 = sum(ce,2);

%%
BinaryY = zeros(size(TestX,1),1);

if 0 %straight up comparison
    for i = 1:size(TestX,1)
        BinaryY(i) = double( ce1(i) > ce0(i) );
    end
else
    
    % hungarian
    testxbig =[TestX0; TestX1];
    [C,IA,IC] = unique(testxbig,'rows','stable');
    
    A = inf+zeros(size(TestX,1));
    
    for jj =1:size(A,2)        
        ind0 = IC(jj);
        ind1 = IC(jj+size(TestX0,1));
        
        A(ind0,jj) = -ce0(jj);
        A(ind1,jj) = -ce1(jj);       
    end
       
    [H,val] = hungarian(A);

    assert(size(H,2)==size(TestX,1));
    for i = 1:size(TestX,1)        
        if maxabs(C(H(i),:), TestX0(i,:))==0
            BinaryY(i) = 0;           
        elseif maxabs(C(H(i),:), TestX1(i,:))==0
            BinaryY(i) = 1;            
        else
            assert(false);        
        end
    end
end


%%
if 1
    filename = 'submit.it.5.31.csv';
    fid = fopen(filename, 'w+');
    for i = 1:size(BinaryY,1)
        fprintf(fid,'%d\n', int32(BinaryY(i)));
    end

    fprintf('\n Finished writing to %s\n', filename);
    fclose(fid);
end
rg;
