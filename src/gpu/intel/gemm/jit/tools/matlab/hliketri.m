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

function [i,j] = hliketri(q,x,y,c,upper)
    i = 0;
    j = 0;
    bl = false;
    if nargin < 4, c = 1; end
    p1 = 1;
    p0 = ~upper;
    triangular = true;
    
    while triangular && min(x,y) > 1
        if (c > 0) == upper
            % y \approx cx
            ih = floor((x+1)/2);
            jh = abs(c)*ih;
            tt1 = jh*(ih+1)/2;
        else
            % x \approx -cy
            jh = floor((y+1)/2);
            ih = abs(c)*jh;
            tt1 = ih*(jh+1)/2;
        end

        if (q < tt1)
            x = ih;
            y = jh;
        else
            if upper
                j = j + jh;
                y = max(y - jh, 0);
                qt = y*ih;
            else
                i = i + ih;
                x = max(x - ih, 0);
                qt = x*jh;
            end
            q = q - tt1;
            if q < qt
                if upper
                    x = ih;
                else
                    y = jh;
                end
                triangular = false;
            else
                q = q - qt;
                if upper
                    x = x - ih;
                    i = i + ih;
                else
                    y = y - jh;
                    j = j + jh;
                end
            end
        end
    end
    
    % note p1 = 1 on entry, but bl = 0
    
    while min(x,y) > 1
        ih = floor((x+1)/2);
        jh = floor((y+1)/2);
        xo = x;
        yo = y;
        p1o = p1;
        
        if bl, q = x*y - q - 1; end
        
        if p0 == 0
            by = q >= x*jh;
                if by, q = x*y - q - 1; end
            y = floor((y + 1 - by)/2);
            bx = q >= ih*y;
                if bx, q = q - ih*y; end
            x = floor((x + 1 - bx)/2);
            bl = by;
            p0 = bitxor(p0, ~bx);
            p1 = bitxor(p1, bitand(by, ~bx));
        else
            bx = q >= ih*y;
                if bx, q = x*y - q - 1; end
            x = floor((x + 1 - bx)/2);
            by = q >= x*jh;
                if by, q = q - x*jh; end
            y = floor((y + 1 - by)/2);
            bl = bx;
            p0 = bitxor(p0, ~by);
            p1 = bitxor(p1, bitand(bx, ~by));
        end
        
        i = i + floor((xo + 1 - p1o)/2) * bitxor(bx, p1o);
        j = j + floor((yo + 1 - p1o)/2) * bitxor(by, p1o);
    end
    
%     while min(x,y) > 1
%         ih = floor(x/2);
%         jh = floor(y/2);
%         xo = x;
%         yo = y;
%         p1o = p1;
%         
%         if bl, q = x*y - q - 1; end
%         
%         if p0 == 0
%             by = q >= x*jh;
%                 if by, q = x*y - q - 1; end
%             y = floor((y + by)/2);
%             bx = q >= ih*y;
%                 if bx, q = q - ih*y; end
%             x = floor((x + bx)/2);
%             bl = by;
%             p0 = bitxor(p0, ~bx);
%             p1 = bitxor(p1, bitand(by, ~bx));
%         else
%             bx = q >= ih*y;
%                 if bx, q = x*y - q - 1; end
%             x = floor((x + bx)/2);
%             by = q >= x*jh;
%                 if by, q = q - x*jh; end
%             y = floor((y + by)/2);
%             bl = bx;
%             p0 = bitxor(p0, ~by);
%             p1 = bitxor(p1, bitand(bx, ~by));
%         end
%         
%         i = i + floor((xo + p1o)/2) * bitxor(bx, p1o);
%         j = j + floor((yo + p1o)/2) * bitxor(by, p1o);
%     end
    i = i + mod(q,x);
    j = j + floor(q/x);
end
