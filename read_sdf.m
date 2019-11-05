%------------------------------------------------------------------------------%
%---------------------------   read sdf function    ---------------------------%
%                                                                              %
% [X,Y,Z] = read_sdf(filename) is a special function that reads the z values   %
% of files with extension .sdf. These files must have a specific configurati-  %
% on, with a certain header and composition of their data. Then the function,  %
% taking into account that, is responsible for properly extracting the z va-   %
% lues.                                                                        % 
%                                                                              %
% INPUTS:                                                                      %
%   -filename:    direct path to the sdf file that you want to read.           %
%                                                                              %
%   >> filename = 'files/test_JP2.sdf';                                        %
%                                                                              %
% OUTPUTS:                                                                     %
%   -X: x coordinates corresponding to each z value in the file. These coordi- %
%       nates consider the number of elements and the scale that the file in-  %
%       dicates in its header.                                                 %
%   -Y: y coordinates corresponding to each z value in the file. These coordi- %
%       nates consider the number of elements and the scale that the file in-  %
%       dicates in its header.                                                 %
%   -Z: z values read from the file. Consider the z scaling specified in the   %
%       header.                                                                %
%                                                                              %
%  NOTE: Now all the size parameters of the polygonal shapes (w and h)         %
%        will be defined by the distribution parameters specified in polyshape %
%                                                                              %
%  Example of usage:                                                           %
%      [Xg,Yg,Zg] = read_sdf(filename); read the file                          %
%      figure(), surf(Xg, Yg, Zg)       make the surface                       %
%                                                                              %
%------------------------------------------------------------------------------%

function [X,Y,Z] = read_sdf(filename)

	% --- read the data from filename
		disp(' ** read the data from filename ...')
		% A = importdata(filename, '%s');
		A = textread(filename, '%s');
	
	% --- parce the parameters
		disp(' ** parce the parameters ...')
		npoints = str2double(A{12});
		nprofiles = str2double(A{15});
		xscale = str2double(A{18});
		yscale = str2double(A{21});
		zscale = str2double(A{24});

	% --- Get the true data
		disp(' ** Get the true data ...');
		% M = cellfun(@str2double, A.textdata(:, 1));
		M = str2double(A(38:end));
		M = M(~isnan(M));

	% --- Get the X, Y, Z values
		disp(' ** Get the X, Y, Z values ...');

		% raw X, Y, Z
		[X, Y] = meshgrid(1:npoints, 1:nprofiles);
		Z = reshape(M, npoints, nprofiles);

		% rescale
		X = X * xscale;
		Y = Y * yscale;
		Z = Z * zscale;

