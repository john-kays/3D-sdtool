%------------------------------------------------------------------------------%
%-------------------------   get features function    -------------------------%
%                                                                              %
% [wc, hc, xp, yp] = get_features(Zsurf, pts) It is a function that determines %
% the size of each of the shapes (widths and heights) distributed on the Zsurf %
% surface, starting from its centroids. The function internally runs an itera- %
% tive clustering algorithm to extract all the points of each sahpe starting   %
% from the centroid. The algorithm iteratively selects point rings from the    %
% centroid, computes the average value (corresponding to the average height of %
% the ring) and then computes the distance of the current ring from the pre-   %
% vious one. The reference or stop criterion is the distance of the first ring %
% with respect to the centroid. If the distance between the current ring and   %
% the previous one is less than that reference, then the grouping ends. Then   %
% the last ring found corresponds to the width of the current pattern under    %
% study. Finally, the width of the shapes is the diameter of that last ring    %
% and the heights are simply the evaluation of Zsurf in the centroid.          %
%                                                                              %
% INPUTS:                                                                      %
%   -Zsurf: M-by-N matrix of Zsurf values that, seen as a surface, contain     %
%           certain patterns or shapes whose size (width and height) vary ac-  %
%           cording to a normal distribution and whose distance between center %
%           to center (pitch) along of the X and Y axes also follows a normal  %
%           distribution                                                       %
%                                                                              %
%    >> [~,~,Zurf] = read_sdf('test_JP2.sdf');                                 %
%                                                                              %
%   -pts: centroid points of each of the patterns in Zsurf                     %
%                                                                              %
% OUTPUTS:                                                                     %
%   -wc: Width of each of the shapes                                           %
%   -hc: Height of each of the shapes                                          %
%   -xp: X coordinate of the centroids of each shape geolocated in a matrix    %
%        of the same dimension as Zsurf (the matrices contain NaN values at    %
%        the other points where there is no centroid).                         %
%   -yp: Y coordinate of the centroids of each shape geolocated in a matrix    %
%        of the same dimension as Zsurf (the matrices contain NaN values at    %
%        the other points where there is no centroid).                         %
%                                                                              %
%  Example of usage:                                                           %
%      [~,~,Zurf] = read_sdf('test_JP2.sdf');                                  %
%                                                                              %
%      % -- 2D surf from zero                                                  %
%      Zsurf = Zsurf - min(Zsurf(:));                                          %
%                                                                              %
%      % -- function that find peaks over 2D signals (centroids)               %
%      pks = FastPeakFind(Z, median(Z));                                       %
%                                                                              %
%      % -- then the centroids are                                             %
%      centers = [pks(1:2:end), pks(2:2:end)];                                 %
%                                                                              %
%      % -- execute our function                                               %
%		   [wc, hc, xp, yp] = get_features(Z, centers);                            %
%                                                                              %
%------------------------------------------------------------------------------%

function [wc, hc, pc] = get_features(Zsurf, pts)

	% --- initialization

		% work size
		s = size(Zsurf);

		% get the surf domain
		[X, Y] = meshgrid( 1:s(2), 1:s(1) );

		% get cluster number
		K = size(pts, 1);

		% create matrices to return each feature
		wc = zeros(K, 1); % pattern widths
		hc = zeros(K, 1); % pattern heights
		pc = zeros(K, 1); % pattern pitch
		% xp = NaN(s);      % coordenate x for geolocatization at the surface
		% yp = NaN(s);      % coordenate y geolocatization at the surface

	% --- sweep for each centroid

	for k = 1:K

		% get the centroid coordenates
		xo = pts(k, 1);
		yo = pts(k, 2);

		% create a mask for clustering
		mask = (X-xo).^2 + (Y-yo).^2;

		% variables to comparison
		udz = 1; % z distance between rings for clustering
		r = 1;   % initial radius size for the rings

		% --- Clustering phase

			while(true)

				% get the current mask (increases according to the radius)
				ind = mask == r^2; % a ring of radio 'r'

				% get the values at the ring
				Z = Zsurf(ind);

				% distance from the height of the centroid
				dz = Zsurf(yo, xo) / mean(Z) - 1;

				% for the first radius value
				if r == 1
					% set the stopping criterio 'ref'
					ref = dz;
					udz = dz;
				% for else  radius values
				else
					% distance between previus and current ring 
					udz = dz - pdz;
				end

				% set the current distance as previus distance for next iteration
				pdz = dz;

				% increase the radio
				r = r + 1;

				% stopping criterion 
				if udz < ref
					% clustering finalized
					break
				end
			end

		% last ind has the width information
		% get the width of the current cluster (pattern)
		[iy, ix] = find(ind);
		wc(k) = mean([ range(iy), range(ix) ]);

		% get the heigth of the current cluster
		hc(k) = Zsurf(yo, xo) - mean(Z(:)) * .8;

		% fill the coordenates in the geolocalization matrices
		% xp(yo, xo) = xo;
		% yp(yo, xo) = yo;

		euc = sqrt( (pts(:, 1) - xo).^2 + (pts(:, 2) - yo).^2 );
		euc = sort(euc);
		pc(k, 1) = euc(2)*.7;
	end