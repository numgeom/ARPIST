function [fs] = spherical_integration(xs, surfs, fun_handle, deg)
% This script integration a function on a mesh of a sphere.
% INPUT:
%   xs      coordinates of vertices on the unit sphere, nx3
%   surfs   connectivity table, nxm
%           We support arbitrary n-gon and a mix of them.
%           If the edge number of a cell is less than the size of the
%           second dimension, fill 0s in the empty position.
%           For example, a mesh contain a triangle and quadrilateral, it
%           may look like:
%           [1, 2, 3, 0; 2, 1, 4, 5]
%           123 is the triangle and 2145 is the quadrilateral.
%   fun_handle
%           matlab function handle of the test function
%           Use @(x)(1) for area integration.
%   deg     degree of ARPIST
%           if deg <= 0, we would use adaptive ARPIST.
% OUTPUT:
%   fs      function values on all the elements, nx1

% Copyright (C) 2022 NumGeom Group at Stony Brook University

if (nargin < 4); deg = -1; end

%% generate quadrature points
% This part could be reused for multiple functions
if (deg > 0)
    % if degree is set up, use the given degree
    [pnts, ws, offset] = compute_sphere_quadrature(xs, surfs, 100, deg);
else
    % if degree is not set up, use our configuration for adaptive ARPIST
    [pnts, ws, offset] = compute_sphere_quadrature(xs, surfs);
end

%% evaluate function values on each element
% This part could be modified to deal with multiple functions
nf = size(surfs, 1);
fs = zeros(nf, 1);

for fid = 1:nf
    
    for pid = offset(fid):offset(fid + 1) - 1
        fs(fid) = fs(fid) + fun_handle(pnts(pid, :)) * ws(pid);
    end
    
end

end
