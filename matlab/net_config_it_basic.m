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

function [net_layers] = net_config_it_basic()
net_layers=[];


L = 1;
net_layers{L}.type = 'convdata';
net_layers{L}.nFilters = 3;
net_layers{L}.nI_grid = 48;
net_layers{L}.nJ_grid = 48;
net_layers{L}.f_dropout = 0.0;

L = L+1;
net_layers{L}.type = 'convc';
net_layers{L}.nVisChannels = 3;
net_layers{L}.nFilters = 16;
net_layers{L}.nVisI = 48;
net_layers{L}.nVisJ = 48;
net_layers{L}.nI_filt = 5;
net_layers{L}.nJ_filt = 5;
net_layers{L}.nPaddingStart = -2;
net_layers{L}.nStride = 1;
net_layers{L}.nI_grid = 48;
net_layers{L}.nJ_grid = 48;
net_layers{L}.nNeuronType = 3;
net_layers{L}.nCPUMode = 0;
net_layers{L}.f_dropout = 0.0;
net_layers{L}.f_wtcost = 0.0004;
net_layers{L}.nPartialSum = 8;
net_layers{L}.ri_mode = 0;
net_layers{L}.ri = 0.01;
net_layers{L}.initB = 0.01;
net_layers{L}.nPrevLayerID = 1;


L = L+1;
net_layers{L}.type = 'convs';
net_layers{L}.nFilters = 16;
net_layers{L}.nStride = 2;
net_layers{L}.nSizeX = 3;
net_layers{L}.nI_sgrid = 24;
net_layers{L}.nJ_sgrid = 24;
net_layers{L}.nNeuronType = 0;
net_layers{L}.f_dropout = 0.0;
net_layers{L}.nPoolingType = 1;
net_layers{L}.nPoolAKMode = 1;
net_layers{L}.nPrevLayerID = L-1;


L = L+1;
net_layers{L}.type = 'convc';
net_layers{L}.nVisChannels = 16;
net_layers{L}.nFilters = 32;
net_layers{L}.nVisI = 24;
net_layers{L}.nVisJ = 24;
net_layers{L}.nI_filt = 5;
net_layers{L}.nJ_filt = 5;
net_layers{L}.nPaddingStart = -2;
net_layers{L}.nStride = 1;
net_layers{L}.nI_grid = 24;
net_layers{L}.nJ_grid = 24;
net_layers{L}.nNeuronType = 3;
net_layers{L}.nCPUMode = 0;
net_layers{L}.f_dropout = 0.0;
net_layers{L}.f_wtcost = 0.0004;
net_layers{L}.nPartialSum = 8;
net_layers{L}.ri_mode = 0;
net_layers{L}.ri = 0.01;
net_layers{L}.initB = 0.01;
net_layers{L}.nPrevLayerID = L-1;


L = L+1;
net_layers{L}.type = 'convs';
net_layers{L}.nFilters = 32;
net_layers{L}.nStride = 2;
net_layers{L}.nSizeX = 3;
net_layers{L}.nI_sgrid = 12;
net_layers{L}.nJ_sgrid = 12;
net_layers{L}.nNeuronType = 0;
net_layers{L}.f_dropout = 0.0;
net_layers{L}.nPoolingType = 0;
net_layers{L}.nPoolAKMode = 1;
net_layers{L}.nPrevLayerID = L-1;

L = L+1;
net_layers{L}.type = 'fc';
net_layers{L}.nV = 12^2*32;
net_layers{L}.nH = 2*512;
net_layers{L}.nNeuronType = 3;
net_layers{L}.f_dropout = 0.5;
net_layers{L}.f_wtcost = .001;
net_layers{L}.f_wt_cons_val = 0;
net_layers{L}.ri_mode = 0;
net_layers{L}.ri = 0.01;
net_layers{L}.initB = 0.01;
net_layers{L}.nPrevLayerID = L-1;


L = L+1;
net_layers{L}.type = 'fc';
net_layers{L}.nV = 2*512;
net_layers{L}.nH = 1024;
net_layers{L}.nNeuronType = 1;
net_layers{L}.fC = 1;
net_layers{L}.f_dropout = 0.0;
net_layers{L}.f_wtcost = .0;
net_layers{L}.f_wt_cons_val = 0;
net_layers{L}.ri_mode = 0;
net_layers{L}.ri = 0.01;
net_layers{L}.initB = 0;
net_layers{L}.nPrevLayerID = L-1;
