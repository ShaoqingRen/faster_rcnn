function tic_toc_print(fmt, varargin)
% Print only after 1 second has passed since the last print. 
% Arguments are the same as for fprintf.

% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2009-2012 Ross Girshick
% 
% This file is part of the voc-releaseX code
% (http://people.cs.uchicago.edu/~rbg/latent/)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------

persistent th;

if isempty(th)
  th = tic();
end

if toc(th) > 1
  fprintf(fmt, varargin{:});
  drawnow;
  th = tic();
end
