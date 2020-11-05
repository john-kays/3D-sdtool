%------------------------------------------------------------------------------%
%-----------------------   get 3D pattern statistics    -----------------------%
%                                                                              %
% [polyshape, distparams, imsize, features] = ...                              %
%     get_3D_pattern_statistics(Zsurf, pattype)                                %
% receives an M-by-N matrix of Zsurf values that, seen as a surface, contain   %
% certain patterns or shapes whose size (width and height) vary according to   %
% a normal distribution and whose distance between center to center (pitch)    %
% along of the X and Y axes also follows a normal distribution. Then the func- %
% tion, also knowing as input the type of patterns (pattype, top hemispheres   %
% for example) contained in Zsurf, locate these patterns and extract their     %
% main statistics. That is, it determines the normal distributions associated  %
% with its size and pitch.                                                     %
%                                                                              %
% INPUTS:                                                                      %
%   -Zsurf: M-by-N matrix Z values read from an .sdf file with the special     %
%           function read_sdf().                                               %
%                                                                              %
%  >> [~,~,Zurf] = read_sdf('test_JP2.sdf');                                   %
%                                                                              %
%  -pattype: string indicating the type of patterns contained in Zsurf (See    %
%            table below)                                                      %
%  >> pattype = 'htop' the patterns are top hemispheres (top curvature)        %
%                                                                              %
% OUTPUTS:                                                                     %
%  -polyshape:    Array of 3 cells defining the pattern of the 3D shapes:      %
%                 - First cell is a string specifying the type of 3D shape de- %
%                   sired (see the table below to know the types of shapes     %
%                   available).                                                %
%                 - Second cell is a vector that can be two or four elements.  %
%                   The two first elements are the mean & standar deviation of %
%                   a pseudo-normal distribution used to define the width of   %
%                   each polygon. The 3rd & 4rd elements (optionals) are the   %
%                   min & max values, respectibily, to truncate the pseudo-    %
%                   normal distribution                                        %
%                 - Third cell is similar to the previous one but to define    %
%                   the height of the shapes                                   %
%  >> polyshape = {'htop', [meanwidth, stdwidth], [meanheight, stdheight]}     %
%                               - hemispheres (top curvature) with             %
%                                 width following ~ N(meanwidth, stdwidth)     %
%                                 height following ~ N(meanheight, stdheight)  %
%                                                                              %
%  -distparams:   Vector of two or four elements where the two first of them   %
%                 specify the mean & standar deviation error of a pseudo-      %
%                 normal distribution used to define the center-to-center dis- %
%                 tance (pitch). The 3rd & 4rd elements (optionals) are the    %
%                 min & max values, respectibily, to truncate the pseudo-      %
%                 normal distribution                                          %
%                                                                              %
%  -imsize:       Vector of two elements specifying the size of the 3D out-    %
%                 put surface, 'img'.                                          %
%                                                                              %
%  -features:    Array of four cells. The first cell contains the widths of    %
%                each extracted shape. The second cell contains the height va- %
%                lues. The last cell is an Nc-by-2 matrix that contains the X  %
%                and Y coordinates of the centroids of the extracted shapes.   %
%                Where Nc is the number of centroids                           %
%                                                                              %
%  Table: pattype options                                                      %
%  +------------+-------------+--------------+---------+-----------------+     %
%  | shape      |  long form  |  short form  |  size   |  description    |     %
%  +------------+-------------+--------------+---------+-----------------+     %
%  +------------+-------------+--------------+---------+-----------------+     %
%  | hemisphere | 'htop'      |  'ht'        | [w, h]  | w: width        |     %
%  | top        |             |              |         | h: height       |     %
%  +------------+-------------+--------------+---------+-----------------+     %
%  | hemisphere | 'hbottom'   |  'hb'        | [w, h]  | w: width        |     %
%  | bottom     |             |              |         | h: height       |     %
%  +------------+-------------+--------------+---------+-----------------+     %
%                                                                              %
%  Example of usage:                                                           %
%      [~,~,Zurf] = read_sdf('test_JP2.sdf');                                  %
%      [polyshape, distparams, imsize, features] = ...                         %
%          get_3D_pattern_statistics(Zsurf, pattype);                          %
%                                                                              %
%      % execute the previus function to replicate the original Zsurf          %
%      [imgp, pitchp, wsp, hsp] = ...                                          %
%          shape_placement_3D_V2(imsize, polyshape, distparams);               %
%      Zp = imgp(:,:,3);                                                       %
%                                                                              %
%      % original 3D surf                                                      %
%      figure(), surf(Z)                                                       %
%      xlabel('X'), ylabel('Y');                                               %
%      xlim([0 imsize(1)])                                                     %
%      ylim([0 imsize(2)])                                                     %
%                                                                              %
%      % replicated 3D surf                                                    %
%      figure(), surf(Zp);                                                     %
%      xlabel('X'), ylabel('Y');                                               %
%      xlim([0 imsize(1)])                                                     %
%      ylim([0 imsize(2)])                                                     %
%                                                                              %
%------------------------------------------------------------------------------%
function [polyshape, distparams, imsize, features] = get_3D_pattern_statistics(Zsurf, pattype)

	% --- initializations

		% add necessary functions
		% functionname = 'get_3D_pattern_statistics.m';
		% functiondir = which(functionname);
		% functiondir = pwd;
		% addpath([functiondir '/utils'])
		Zsurf = min(10e-5, Zsurf);

		% 2D signal from zero
		Z = Zsurf - min(Zsurf(:));

		% work size
		ly = size(Z, 1);
		lx = size(Z, 2);

	% --- get the centroids

		% function that find peaks over 2D signals
		pks = FastPeakFind(Z, median(median(Z)));

		% then the centroids are
		centers = [pks(1:2:end), pks(2:2:end)];

	% --- get the features from centroids

		% function that computes the features
		% [wc, hc, xp, yp] = get_features(Z, centers);
		[wc, hc, pc] = get_features(Z, centers);

	% --- get the statistics

		% -- center-to-center pitch
			% wcmean = mean(wc);
			% wcstd = std(wc);
			% wclow = floor(wcmean - 0);
			% nnx = floor(lx/wclow);
			% nny = floor(ly/wclow);

			% pitch = [];
			% for i = 1:nny
			% 	st = (i-1) * wclow + 1;
			% 	en = min(ly, (i  ) * wclow);
			% 	xpc = xp(st:en, :);
			% 	d = diff(sort( xpc(~isnan(xpc)) ));
			% 	d = d(d~=0);
			% 	pitch = [pitch; d];
			% end
			% for i = 1:nnx
			% 	st = (i-1) * wclow + 1;
			% 	en = min(lx, (i  ) * wclow);
			% 	ypc = yp(:, st:en);
			% 	d = diff(sort( ypc(~isnan(ypc)) ));
			% 	d = d(d~=0);
			% 	pitch = [pitch; d];
			% end

			% pmean = mean(pitch);
			% pstd = std(pitch);
			pmean = mean(pc);
			pstd = std(pc);
			% pmin = pmean - pstd;
			% pmax = pmean + pstd;
			pmin = min(pc);
			pmax = max(pc);

		% -- width
			wmean = mean(wc);
			wstd = std(wc);
			% wmin = wmean - wstd;
			% wmax = wmean + wstd;
			wmin = min(wc);
			wmax = max(wc);

		% -- height
			hmean = mean(hc);
			hstd = std(hc);
			% hmin = hmean - hstd;
			% hmax = hmean + hstd;
			hmin = min(hc); %hmean - hstd;
			hmax = max(hc); %hmean + hstd;


	% --- output variables

		% polyshape = {pattype, [wmean, wstd], [hmean, hstd]};
		% distparams = [pmean, pstd];
		polyshape = {pattype, [wmean, wstd, wmin, wmax], [hmean, hstd, hmin, hmax]};
		distparams = [pmean, pstd, pmin, pmax];
		imsize = [ly, lx];
		features = {wc, hc, pc, centers};