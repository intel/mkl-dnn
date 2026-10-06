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

%%
m = 12;
n = 8;
wg = m * n;
s = 4;
[I,J]=arrayfun(@boustrophedon, 0:(wg-1), m*ones(1,wg), n*ones(1,wg), s*ones(1,wg));
M = [I;J];
walkplot(M)

%%
m = 8;
n = 8;
c = 1;
upper = 0;
if (c > 0)
    if upper
        wg = m*n + (m*c*(1-m))/2;
    else
        wg = m*n + (n*c*(1-n))/2;
    end
else
    if upper
        wg = (-c*n*(n-1))/2 + m;
    else
        wg = (-c*m*(m-1))/2 + n;
    end
end
[I,J]=arrayfun(@hliketri, 0:(wg-1), m*ones(1,wg), n*ones(1,wg), c*ones(1,wg), upper*ones(1,wg));
M = [I;J];
walkplot(M)

%%
m = 7;
n = 11;
bias = 1;
[I,J]=arrayfun(@hlike, 0:(m*n-1), m*ones(1,m*n), n*ones(1,m*n), bias*ones(1,m*n));
M = [I;J];
walkplot(M)
