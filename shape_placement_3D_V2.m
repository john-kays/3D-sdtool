%------------------------------------------------------------------------------%
%----------------------   shape placement 3D function    ----------------------%
%                                                                              %
% 'img = shape_placement_3D (imsize, polyshape, distparams)'                   % 
% creates the 3D surface 'img' in point cloud format. The 'img' is returned as % 
% a 3D matrix where each layer in the 3rd dimesion represent the points in     % 
% each plane X,Y,Z. The surface size is defined by the 'imsize' parameter and  %
% it has 3D shapes specified by 'polyshape' placed along it. The size of the   %
% 3D shapes varies according to a normal distribution defined by the distribu- % 
% tion parameters given by the polyshape input argument. The center-to-center  % 
% distance along the x and y directions between the polygonal shapes follows a %
% pseudonormal distribution defined by distparams. The surface 'img' can be    %
% visualized on a 3D way by using the 'surf' & 'imagesc' functions.            %
%                                                                              %
%  -imsize:       Vector of two elements specifying the size of the 3D out-    %
%                 put surface, 'img'.                                          %
%                                                                              %
%  >> imsize = [m, n]  % 'img' surface with  m units on x and n units on y     %
%                                                                              %
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
%  >> distparams = [mean, std]                                                 %
%                                                                              %
%  Table: polyshape options                                                    %
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
%  NOTE: Now all the size parameters of the polygonal shapes (w and h)         %
%        will be defined by the distribution parameters specified in polyshape %
%                                                                              %
%  Example of usage:                                                           %
%      polyshape = {'ht', [12, 4], [2, .5]};                                   %
%      imsize = [100, 100]                                                     %
%      distparams = [20, 5]                                                    %
%      img = shape_placement_3D(imsize, polyshape, distparams)                 %
%      X = img(:,:,1);                                                         %
%      Y = img(:,:,2);                                                         %
%      Z = img(:,:,3);                                                         %
%      Z = Z + 30;                                                             %
%      s = surf(X,Y,Z);                                                        %
%      hold on                                                                 %
%      imagesc(X(:), Y(:), Z)                                                  %
%      zlim([-10 h+10])                                                        %
%      view(45, 20)                                                            %
%------------------------------------------------------------------------------%
function [img, pr, wsp, hsp] = shape_placement_3D_V2(imsize, polyshape, distparams)

%--- Initializations
	
	% Getting polyrotation
	% if ~exist('polyrotation')
	% 	polyrotation = 'on';
	% end

	% Estimating the number of shapes per dimension
	pmean = distparams(1);
	pmean_div2 = pmean / 2;
	nshapes = [ceil(imsize(1)/pmean), ceil(imsize(2)/pmean)] + 0;

%--- Get the piths for all the shapes

	% Getting it from a pseudo normally distribution defined by
	%	the pmean & pstd
	pmean = distparams(1);
	pstd = distparams(2);
	prr = pstd * randn([nshapes,2]) + pmean;
	pitchs = (pstd * .5) * randn([nshapes.^2,2]);
	pitchs = pitchs(1:nshapes(1), 1:nshapes(2), :);
	% pitchs = pitchs(randperm(nshapes(1)^2,nshapes(1)), randperm(nshapes(2)^2,nshapes(2)), :);
	if length(distparams) > 2
		pitchs = max((distparams(3)-pmean) * .5, min((distparams(4)-pmean) * .5, pitchs));
	end

%--- Get the size for all the shapes

	% Getting it from a pseudo normally distribution defined by
	%	the meansize & stdsize ditribution parameters

	% wights
	wp = polyshape{2};
	wmean = wp(1);
	wstd = wp(2);
	ws = wstd.*randn([nshapes.^2,2]) + wmean;
	ws = ws(1:nshapes(1), 1:nshapes(2), :);
	wsp{1} = ws;
	if length(wp) > 2
		ws = max(wp(3), min(wp(4), ws));
		wsp{2} = ws;
	end

	% heigts
	hp = polyshape{3};
	hmean = hp(1);
	hstd = hp(2);
	hs = hstd.*randn([nshapes.^2,2]) + hmean;
	hs = hs(1:nshapes(1), 1:nshapes(2), :);
	hsp{1} = hs;
	if length(hp) > 2
		hs = max(hp(3), min(hp(4), hs));
		hsp{2} = hs;
	end

%--- Get the angles for the rotation

	% Getting uniform random rotation
	% if strcmpi(polyrotation, 'on')
	% 	rotations = 360*rand([nshapes]);
	% end

%--- Placement radonming the shape at the image im

	% Creating a variable for counting the shapes
	% ishape = 1;

	% Creating the output 3D image as a point cloud
	[X,Y] = meshgrid(1:imsize(1), 1:imsize(2));
	% Z = zeros(size(X))
	img = zeros( [size(X), 3] );
	img(:,:,1) = X;
	img(:,:,2) = Y;
	img(:,:,3) = 0.0;

	xo = zeros(1, nshapes(1));
	xprev = xo;
	yo = 0;


	% Sweep over whole image
	%	loop extern to control the sweep at x direction
	for j = 1:nshapes(2)

		% Variable for counting the shapes at the current column
		% term = 1;
		% ishape = y;

		%	loop intern to control the sweep at y direction

		for i = 1:nshapes(1)

			% Getting the current dx & dy pitchs
			% if i == 1
			% 	% xo = -pitchs(end, j, 1)*.5 + pitchs(i, j, 1);
			% 	xo = pitchs(i, j, 1) * .5;
			% else
			% 	xo = xo + pitchs(i, j, 1);
			% end
			% if j == 1
			% 	% yo(i) = -pitchs(i, end, 2)*.5 + pitchs(i, j, 2);
			% 	yo(i) = pitchs(i, j, 2) * .5;
			% else
			% 	yo(i) = yo(i) + pitchs(i, j, 2);
			% end
			xprev(i) = xo(i);
			yprev = yo;
			xo(i) = (j - 1) * pmean + pmean_div2 + pitchs(i, j, 1);
			yo = (i - 1) * pmean + pmean_div2 + pitchs(i, j, 2);
			if i > 1
				py(i-1, j) = yo - yprev;
			end
			if j > 1
				px(j-1, i) = xo(i) - xprev(i);
			end

			% Getting the sizes
			w = ws(i, j);
			h = hs(i, j);

			% Getting the grid domain
			% x direction
			js = max(1, floor(xo(i)) - floor(w));
			je = min(imsize(2), floor(xo(i)) + floor(w));
			% ydirection
			is = max(1, floor(yo) - floor(w));
			ie = min(imsize(1), floor(yo) + floor(w));

			Xd = X(is:ie, js:je);
			Yd = Y(is:ie, js:je);
			
			% Getting the dynaminc size mask
			Zd = getshape( polyshape{1}, Xd, Yd, ...
				w, h, xo(i), yo );

			% Making the polygonal rotation
			% if strcmpi(polyrotation, 'on')
			% 	polymask = imrotate(polymask, rotations(x,y), 'bilinear');
			% end

			% Defining dynamic variables to control the sweep
			% Xo = (j - 1) * pmean + pmean_div2;
			% Yo = (i - 1) * pmean + pmean_div2;
			% Zo = 0;

			% Getting the current layer
			% ilayer = mod(ishape-1, N) + 1;

			% Getting the img's indexers
			% is = (i - 1) * pmean + 1;
			% js = (j - 1) * pmean + 1;
			% ie = is + pmean - 1;
			% je = js + pmean - 1;

			% Placement the shape at the corresponding layer
			img(is:ie, js:je, 1) = Xd;
			img(is:ie, js:je, 2) = Yd;
			img(is:ie, js:je, 3) = (img(is:ie, js:je, 3) + Zd);

			% Counting the ishape variable
			% ishape = ishape + term;
		end
	end

	% [X,Y] = meshgrid(1:1:(nshapes+0)*pmean);
	% img(:,:,1) = X;
	% img(:,:,2) = Y;

	% Getting the internal 3D image
	% is = pmean + 1;
	% js = pmean + 1;
	% ie = pmean * nshapes(1) - 1;
	% je = pmean * nshapes(2) - 1;
	% img = img(is:ie, js:je, :);

	% Creating the point cloud file
	% ptc = pointCloud(img);
	% pcwrite(ptc,'shape_placement_3D.pcd');
	pr(:,:,1) = px;
	pr(:,:,2) = py;

end

function [Z] = getshape(shape, X, Y, w, h, xo, yo)
	if strcmpi(shape, 'ht') || strcmpi(shape, 'htop')
		[X,Y,Z] = hemisphere(X, Y, w, h, xo, yo);
	elseif strcmpi(shape, 'hb') || strcmpi(shape, 'hbottom')
		[X,Y,Z] = hemisphere(X, Y, w, h, xo, yo);
		Z = -Z;
	elseif strcmpi(shape, 'pt') || strcmpi(shape, 'cylinder')
		[X,Y,Z] = hemisphere(X, Y, w, h, xo, yo);
	elseif strcmpi(shape, 'pb') || strcmpi(shape, 'cone')
		[X,Y,Z] = hemisphere(X, Y, w, h, xo, yo);
	end
end

function [X,Y,Z] = hemisphere(X, Y, w, h, xo, yo)
	denf = .45;
	R = w / 2;
	Z = h*exp(-( ( (X-xo).^2/(denf*R^2) ) + ( (Y-yo).^2/(denf*R^2) ) ));

	if max(Z(:)) > 3
		max(Z(:))
	end
	
	% Z = (R^2 - (X-xo).^2 - (Y-yo).^2);
	% Z(Z<0) = 0;
	% Z = sqrt(Z);
	% Z = (Z/R) * h;
end