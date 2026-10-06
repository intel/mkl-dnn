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

function [i,j] = boustrophedon(q,x,y,s0)
    % In real implementation,
    % if i0 > |thresh|, s <- s0+1/s0-1 depending on sgn(thresh)
    % clamp s to not exceed box
    
    s = s0;
    is0 = floor(q/(s*y));       % or floor((q-thresh*y)/(s*y))
    i0 = is0*s;                 % or is0*s + thresh
    q = q - i0*y;
    
    if q < s*s && is0 > 0
        if (q < s*(s - 1)/2)
            q = s*(s - 1)/2 - 1 - q;
            ii = floor(sqrt(2*q + 1/4) + 1/2);
            jj = q - ii*(ii-1)/2;
        else
            q = q - s*(s - 1)/2;
            jj = floor(sqrt(2*q + 1/4) - 1/2);
            ii = q - jj*(jj+1)/2;
        end
        ii = s - 1 - ii;
    elseif q >= s*(y - s) && (i0 < x - s)
        q = q - s*(y - s);
        if (q < s*(s + 1)/2)
            q = s*(s + 1)/2 - 1 - q;
            jj = floor(sqrt(2*q + 1/4) - 1/2);
            ii = q - jj*(jj+1)/2;
        else
            q = q - s*(s + 1)/2;
            ii = floor(sqrt(2*q + 1/4) + 1/2);
            jj = q - ii*(ii-1)/2;
        end
        jj = y - 1 - jj;
    else
        ii = mod(q, s);
        jj = floor(q / s);
    end
    if (bitand(is0, 1))
        jj = y - 1 - jj;
    end
    i = i0 + ii;
    j = jj;
end
