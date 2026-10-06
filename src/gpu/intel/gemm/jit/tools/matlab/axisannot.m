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

function obj = axisannot(type,x,y)
obj = annotation(type);
obj.Parent = gca;
obj.Position = [x(1) y(1) x(2)-x(1) y(2)-y(1)];
if (strcmp(type, 'arrow'))
    obj.HeadLength = 7;
    obj.HeadWidth = 7;
end
end
