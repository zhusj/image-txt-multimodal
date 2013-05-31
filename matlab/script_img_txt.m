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

global DataX;
global DataY;

global dict2;
global freq;
global bad;

tic;
load esp_data.mat;
toc;

for bb = 1:length(bad)    
    DataX(bad(bb),:) =DataX(bad(bb)-1,:);
    DataY(bad(bb),:) =DataY(bad(bb)-1,:);    
end

%%
addpath('../cuda_ut/bin');
hp =[];
hp.MaxIters = 50000;
hp.start_rate = 0.001;
hp.HalfLife = 200000;
hp.momen = 0.9;
hp.noise1 = .1; %1
hp.noise2 = 0.05; %.5
hp.nSamples = 100;
hp.momen_init = 5000;
hp.RM_MEAN_STD = 1;
%hp.D = 70;  %1875
hp.D = 48^2*3;
hp.Dy = 1024;

hp.net_layers = net_config_it_basic();

hp.randseeds = 1234;
hp.normalseeds = 1234;
hp.nSPLIT = 5;

gg;
[ cv_average, cv_models ] = ...
    img_txt_cv( ...
     hp.nSPLIT, hp.randseeds, hp.normalseeds, hp);
rg;

cv_average
mean(cv_average)

if exist('CC', 'var')
    hp.CC = CC;
end

save 'model.it.5.31.mat' cv_average cv_models hp;                          
                            