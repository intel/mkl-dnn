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

function [i,j] = hlike(q,x,y,bias)
    i = 0;
    j = 0;
    bl = false;
    if nargin < 4, bias = 1; end
    p1 = 0;
    p0 = 0;
    
    if min(x,y) > 1
        r = y/(x*bias);
        if r >= 1
            yd = ceil(y / round(2*r));
            by = floor(q / (x*yd));
            by1 = bitand(by, 1);
            q = q - by*x*yd;
            j = by * yd;
            y = min(yd, y - j);
            ih = floor((x + 1)/2);
            xd = floor((x + 1 - by1)/2);
            bx = q >= xd*y;
            if bx, q = q - xd*y; end
            if by1, bx = ~bx; end
            i = bx * ih;
            x = floor((x + 1 - bx) / 2);
            p0 = ~bx;
            p1 = ~bx * by1;
        else
            xd = ceil(x / round(2/r));
            bx = floor(q / (y*xd));
            bx1 = bitand(bx, 1);
            q = q - bx*y*xd;
            i = bx * xd;
            x = min(xd, x - i);
            jh = floor(y/2);
            yd = floor((y + bx1)/2);
            by = q >= x*yd;
            if by, q = q - x*yd; end
            if bx1, by = ~by; end
            j = by * jh;
            y = floor((y + by) / 2);
            p0 = by;
            p1 = ~by * bx1;
        end
    end

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
