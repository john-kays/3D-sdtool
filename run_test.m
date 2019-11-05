close all; clc

% --- parameters

	filename = 'inputs/test_JP2.sdf';
	pattype = 'ht';

% --- load the dataset
	
	[Xg,Yg,Zg] = read_sdf(filename); % read the file
	no = 450; nx = 650; ny = 650;    % work domain
	Xsurf = Xg(no:ny, no:nx);
	Ysurf = Yg(no:ny, no:nx);
	Zsurf = Zg(no:ny, no:nx);

% --- get the 3D pattern statistics
	
	% execute the main function
	[polyshape, distparams, imsize, features] = get_3D_pattern_statistics(Zsurf, pattype);

% --- replicate the original 3D surface
	
	% execute the previus function
	[imgp, pitchp, wsp, hsp] = shape_placement_3D_V2(imsize, polyshape, distparams);

	Zp = imgp(:,:,3);

% --- plot the surfaces
	
	% original 3D surf
	figure(), surf(Zurf)
	% hold on, plot3(p(:, 1), p(:, 2), p(:, 3), 'g*')
	xlabel('X'), ylabel('Y');
	xlim([0 imsize(1)])
	ylim([0 imsize(2)])
	zlim([-1e-7 4e-7])

	figure(), surf(Zp);
	xlabel('X'), ylabel('Y');
	xlim([0 imsize(1)])
	ylim([0 imsize(2)])
	zlim([-1e-7 4e-7])
