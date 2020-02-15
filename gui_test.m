% - open the sdf files
    
    [fn, fp] = uigetfile('*.sdf','Select the INPUT DATA FILE(s)','MultiSelect','on');

    if ~iscell(fn)
    	fn1 = fn;
    	fn = {};
    	fn{1} = fn1;
    end
    nfiles = length(fn);

% - get the 3D pattern statistics
	
	width_min = [];
	width_max = [];
	width_avg = [];
	width_std = [];

	height_min = [];
	height_max = [];
	height_avg = [];
	height_std = [];

	pitch_min = [];
	pitch_max = [];
	pitch_avg = [];
	pitch_std = [];

	filenames = {};

    for ifile = 1:nfiles
    	filename = fn{ifile};
    	ffile = fullfile(fp, filename);

    	if (ifile == 1)
    		fw = waitbar(0, ['processing ', replace(filename, '_', '\_')]);
    	else
    		vervose = sprintf('processing %s %.1f %%', replace(filename, '_', '\_'), perc * 100);
    		waitbar(perc, fw, vervose);
    	end

    	[~, ~, Zg] = read_sdf(ffile);   % read the file

    	no = 450; nx = 650; ny = 650;    % define a work domain
  		Zsurf = Zg(no:ny, no:nx);

		% parameters
		pattype = 'htop'; % We knows that the pattern type is hemisphere top

		% execute the main function
		[polyshape, distparams, imsize, features] = get_3D_pattern_statistics(Zsurf, pattype);

		width_min(ifile,1) = min(features{1}(:));
		width_max(ifile,1) = max(features{1}(:));
		width_avg(ifile,1) = mean(features{1}(:));
		width_std(ifile,1) = std(features{1}(:));

		height_min(ifile,1) = min(features{2}(:));
		height_max(ifile,1) = max(features{2}(:));
		height_avg(ifile,1) = mean(features{2}(:));
		height_std(ifile,1) = std(features{2}(:));

		pitch_min(ifile,1) = min(features{3}(:));
		pitch_max(ifile,1) = max(features{3}(:));
		pitch_avg(ifile,1) = mean(features{3}(:));
		pitch_std(ifile,1) = std(features{3}(:));

		filenames{ifile,1} = filename;

		perc = ifile/nfiles;
		% vervose = sprintf('processing %s %.1f %%', replace(filename, '_', '\_'), perc * 100);
		% waitbar(perc, fw, vervose);
    end

    close(fw);

    % create the table array
    T = table(filenames, ...
    		width_min, width_max, width_avg, width_std, ...
			height_min, height_max, height_avg, height_std, ...
			pitch_min, pitch_max, pitch_avg, pitch_std);

    % select the output filename
    [fn, fp] = uiputfile('*.csv');
    outfile = fullfile(fp, fn);

    % save output file
    writetable(T, outfile, 'Delimiter', ',', 'QuoteStrings', true);