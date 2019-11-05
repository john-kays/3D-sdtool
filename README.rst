######################################
3D-sdtool - 3D Shape Distribution Tool
######################################

:Authors: John Kay, Gilberto Galvis
:Email: john.kays2020@gmail.com, galvisgilberto@gmail.com
:Version: $revision: 0.1.1 $

This project provides a toolkit for handling 3D shapes distributed in a three-dimensional space.

Requirements
------------

- MatLab: The interface tool is based on MatLab, so it is required to have MatLab installed

Installation
------------

- Clone this repository on your machine either using http or ssh

Usage
-----

We really have two tools, which define the approach to using the toolkit: 1) 3D image generation and 2) 3D scenario reconstruction

3D image generation
===================

Having the shape_placement_3D function we can generate a cloud of points that build a series of randomly distributed 3D shapes throughout the three-dimensional space.

Here is an example that can be executed using a MatLab script or also directly in the MatLab Command Window.

.. code:: matlab
	
	clc; close all; clear all;

	% parameters
	polyshape = {'ht', [12, 4, 8, 16], [2, .5, .5, 2.5]};
	imsize = [200, 200];
	distparams = [14, 8, 6, 22];

	% run the function
	[img, pitchs, ws, hs] = shape_placement_3D(imsize, polyshape, distparams);

Please feel free to change the parameters to the values you want

Notes
+++++

- If you need to know in detail about the input parameters of the function, please open the file ``polygon_shape_placement.m``. In that file you can see the source code as well as the explanatory documentation of the function

- Please feel free to review the report.pdf document where we show more examples of this tool.

