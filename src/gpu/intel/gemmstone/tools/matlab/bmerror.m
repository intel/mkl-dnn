%===============================================================================
% Copyright 2021 Intel Corporation
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
%===============================================================================

info = readtable('..\info.txt');
precision = info.Precision{1};
layout = info.Layout{1};
m = info.m(1);
n = info.n(1);
ld = info.ld(1);
is_complex = false;

switch precision
    case 'S'
        mprecision = 'single';
    case 'D'
        mprecision = 'double';
    case 'C'
        mprecision = 'single';
        is_complex = true;
    case 'Z'
        mprecision = 'double';
        is_complex = true;
    case 'I'
        mprecision = 'int32';
    case 'W'
        mprecision = 'int16';
    case 'O'
        mprecision = 'int8';
    case 'o'
        mprecision = 'uint8';
end

msz = [m n];
if layout == 'T'
    msz = fliplr(msz);
end
real_msz = msz;
msz(1) = ld;
if is_complex
    msz(1) = msz(1) * 2;
    real_msz(1) = real_msz(1) * 2;
end
fid = fopen('..\computed.bin');
M = fread(fid, msz, mprecision);
fclose(fid);
fid = fopen('..\ref.bin');
Mref = fread(fid, msz, mprecision);
fclose(fid);
M = M(1:real_msz(1), 1:real_msz(2));
Mref = Mref(1:real_msz(1), 1:real_msz(2));
if is_complex
    M = complex(M(1:2:end,:), M(2:2:end,:));
    Mref = complex(Mref(1:2:end,:), Mref(2:2:end,:));
end
if layout == 'T'
    M = M.';
    Mref = Mref.';
end

if is_complex
    iM = abs(M);
    iMref = abs(Mref);
else
    iM = M;
    iMref = Mref;
end

flat = [iM(:).' iMref(:).'];
mm = [min(flat) max(flat)];
if mm(1) == mm(2)
    mm = mm + [-1 1];
end
ax1 = subplot(2,2,1);
imagesc(iM);
caxis(mm);
colormap jet
colorbar
title Computed
axis square

ax2 = subplot(2,2,2);
imagesc(iMref);
caxis(mm);
colormap jet
colorbar
title Reference
axis square

ax3 = subplot(2,2,3:4);
spy(~(abs(M - Mref) <= 1e-4 * (max(abs(Mref(:))))));

linkaxes([ax1 ax2 ax3])
