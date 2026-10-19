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

function walkplot(M)
    if (size(M, 1) > 2)
        M = M.';
    end
    x = M(1,:);
    y = M(2,:);
    x = x / gcdm(x);
    y = y / gcdm(y);
    l = length(x);
    s = 0.1;
    t = 0.4;
    scatter(x, y, 'filled');
    hold all
    for i = 1:l-1
        x1 = x(i); x2 = x(i+1);
        y1 = y(i); y2 = y(i+1);
        if (mod(i, 4) == 1)
            s_arrow = 'arrow';
        else
            s_arrow = 'line';
        end
        
        if (x2 - x1 == 1) && (y2 == y1)
            axisannot(s_arrow, [x1+s x2-s], [y1 y2]);
        elseif (x2 - x1 == -1) && (y2 == y1)
            axisannot(s_arrow, [x1-s x2+s], [y1 y2]);
        elseif (y2 - y1 == 1) && (x2 == x1)
            axisannot(s_arrow, [x1 x2], [y1+s y2-s]);
        elseif (y2 - y1 == -1) && (x2 == x1)
            axisannot(s_arrow, [x1 x2], [y1-s y2+s]);
        elseif (x2 ~= x1) && (y2 ~= y1)
            axisannot('arrow', [x1+sign(x2-x1)*s x2+sign(x1-x2)*s]...
                             , [y1+sign(y2-y1)*s y2+sign(y1-y2)*s]);
        elseif (x2 ~= x1) && (y2 == y1)
            axisannot('line',  [x1+sign(x2-x1)*s x1+sign(x2-x1)*t]...
                            ,  [y1+s             y1+t]);
            axisannot('arrow', [x1+sign(x2-x1)*t x2+sign(x1-x2)*t]...
                             , [y1+t             y1+t]);
            axisannot('line',  [x2+sign(x1-x2)*t x2+sign(x1-x2)*s]...
                            ,  [y1+t             y1+s]);
        elseif (x2 == x1) && (y2 ~= y1)
            axisannot('line',  [x1+s             x1+t]...
                            ,  [y1+sign(y2-y1)*s y1+sign(y2-y1)*t]);
            axisannot('arrow', [x1+t             x1+t]...
                             , [y1+sign(y2-y1)*t y2+sign(y1-y2)*t]);
            axisannot('line',  [x1+t             x1+s]...
                            ,  [y2+sign(y1-y2)*t y2+sign(y1-y2)*s]);
        end
    end
    hold off
    axis equal
    xlim([min(x)-1 max(x)+1]);
    ylim([min(y)-1 max(y)+1]);
end
