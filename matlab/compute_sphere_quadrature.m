function [pnts, ws, offset] = compute_sphere_quadrature(xs, surfs, ...
    h1, deg1, h2, deg2)
% This function is used to generate quadrature points and corresponding
% weights on a sphere.
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
%   h1      the smaller relative threshold.
%           If the maximum edge length of a cell is lower than radius*h1,
%           we would apply degree-(deg1) quadrature rule for this cell.
%           (default is 0.004)
%   deg1    the smaller degree (default is 4)
%   h2      the larger relative threshold.
%           We would split a cell until the maximum edge length of the sub
%           cells is lower than radius*h2.
%           Once the maximum edge length locates
%           within [h1, h2), we would apply degree-(deg2) quadrature rule
%           for this cell. (default is 0.05)
%   deg2    the larger degree (default is 8)
%
% OUTPUT:
%   pnts    coordinates of quadrature points, nx3
%   ws      corresponding weights for quadrature points, nx1
%   offset  offset for the first quadrature points in each element, nx1
%           All the quadrature point ids for element fid would be
%           offset(fid):offset(fid+1)-1

if (nargin < 3); h1 = 0.004; end
if (nargin < 4); deg1 = int32(4); end
if (nargin < 5); h2 = 0.05; end
if (nargin < 6); deg2 = int32(8); end

%% radius of the sphere
r = norm(xs(1, :));

for vid = 2:size(xs, 1)
    assert(abs(norm(xs(vid, :)) - r) < 2e-6, 'The input mesh is not a sphere');
end

%% Initialization
index = 0;
nf = size(surfs, 1);
max_nv = max(1e6, nf * 6);

pnts = zeros(max_nv, 3);
ws = zeros(max_nv, 1);
nv_surf = size(surfs, 2);
offset = zeros(nf + 1, 1, 'int32');

% go through all the faces
for fid = 1:nf
    offset(fid) = index + 1;
    nhe = nv_surf;
    while (surfs(fid, nhe) <= 0); nhe = nhe - 1; end
    if (nhe < 3); continue; end
    
    % split each element into several spherical triangles
    for j = 2:nhe - 1
        lvids = [1, j, j + 1];
        pnts_tri = xs(surfs(fid, lvids), :) / r;
        h = max_edge_length(pnts_tri);
        
        % generate quadrature points
        if (h < h1)
            [pnts, ws, index] = quadrature_sphere_tri( ...
                pnts_tri, [1, 2, 3], deg1, pnts, ws, index);
        elseif (h < h2)
            [pnts, ws, index] = quadrature_sphere_tri( ...
                pnts_tri, [1, 2, 3], deg2, pnts, ws, index);
        else
            [pnts, ws, index] = quadrature_sphere_tri_split( ...
                pnts_tri, [1, 2, 3], h2, deg2, pnts, ws, index);
        end
        
    end
    
end

pnts = r * pnts(1:index, :);
ws = (r * r) * ws(1:index, :);
offset(nf + 1) = index + 1;
end

function h = max_edge_length(xs)
h = max([norm(xs(1, :) - xs(2, :)), norm(xs(2, :) - xs(3, :)), norm(xs(1, :) - xs(3, :))]);
coder.inline('always');
end

function [pnts, ws, index] = quadrature_sphere_tri( ...
    xs, surfs, deg, pnts, ws, index)
% find cell averaged value for test function f on sphere
% of a mixed mesh.

nf = size(surfs, 1);

[ws_q, cs, nqp] = fe2_quadrule(deg);
cs = [ones(nqp, 1) - cs(:, 1) - cs(:, 2), cs];

if (index + nf * nqp > length(ws))
    n_add = length(ws) + nf * nqp;
    ws = [ws; coder.nullcopy(zeros(n_add, 1))];
    pnts = [pnts; coder.nullcopy(zeros(n_add, 3))];
end

for fid = 1:nf
    pnts_vor = xs(surfs(fid, :), :);
    
    %absulute value of triple product of x1, x2, x3.
    tri_pro = abs(dot(pnts_vor(1, :), cross(pnts_vor(2, :), pnts_vor(3, :))));
    %global coordinate of quadrature points on triangle x1x2x3
    pnts_q = cs * pnts_vor;
    
    for k = 1:nqp
        nrm_q = norm(pnts_q(k, :));
        index = index + 1;
        %project quadrature points on sphere
        pnts(index, :) = pnts_q(k, :) / nrm_q;
        %weights x Jacobi
        ws(index) = ws_q(k) * tri_pro / (nrm_q^3);
    end
    
end

end

function ii = next_leid(ii, n)

coder.inline('always');

if (ii == n)
    ii = int32(1);
else
    ii = ii + 1;
end

end

function [pnts, ws, index] = quadrature_sphere_tri_split( ...
    xs, surfs, tol, deg, pnts, ws, index)
% find cell averaged value for test function f on sphere
% of a mixed mesh.

nf = size(surfs, 1);

h = max_edge_length(xs(surfs(1, :), :));

if (h > tol)
    %split one element
    surf_fid = [1, 4, 6; 6, 4, 5; 5, 4, 2; 6, 5, 3];
    pnts_vor = zeros(6, 3);
    
    for fid = 1:nf
        pnts_vor(1:3, :) = xs(surfs(fid, :), :);
        
        % insert points
        for j = 1:3
            index_local = j + 3;
            pnts_vor(index_local, :) = (pnts_vor(j, :) + pnts_vor(next_leid(j, 3), :)) / 2;
            pnts_vor(index_local, :) = pnts_vor(index_local, :) / norm(pnts_vor(index_local, :));
        end
        
        [pnts, ws, index] = quadrature_sphere_tri_split( ...
            pnts_vor, surf_fid, tol, deg, pnts, ws, index);
    end
    
else
    [pnts, ws, index] = quadrature_sphere_tri( ...
        xs, surfs, deg, pnts, ws, index);
end

end

function [ws, cs, nqp] = fe2_quadrule(degree)
% Obtain quadrature points of a 2-D element.
%

if degree <= 1
    nqp = int32(1);
    
    ws = 0.5;
    
    cs = [0.3333333333333333333333333333333, 0.3333333333333333333333333333333];
    
elseif degree <= 2
    nqp = int32(3);
    
    ws = [0.1666666666666666666666666666666;
        0.1666666666666666666666666666666;
        0.1666666666666666666666666666666];
    
    cs = [0.6666666666666666666666666666666, 0.1666666666666666666666666666666;
        0.1666666666666666666666666666666, 0.6666666666666666666666666666666;
        0.1666666666666666666666666666666, 0.1666666666666666666666666666666];
    
elseif degree <= 4
    nqp = int32(6);
    
    ws = [0.0549758718276609338191631624501052;
        0.0549758718276609338191631624501052;
        0.0549758718276609338191631624501052;
        0.111690794839005732847503504216561;
        0.111690794839005732847503504216561;
        0.111690794839005732847503504216561];
    cs = [0.0915762135097707434595714634022015, 0.0915762135097707434595714634022015;
        0.8168475729804585130808570731956, 0.0915762135097707434595714634022015;
        0.0915762135097707434595714634022015, 0.8168475729804585130808570731956;
        0.445948490915964886318329253883051, 0.445948490915964886318329253883051;
        0.1081030181680702273633414922339, 0.445948490915964886318329253883051;
        0.445948490915964886318329253883051, 0.1081030181680702273633414922339];
elseif degree <= 5
    nqp = int32(7);
    cs = [0.3333333333333333333333333333333, 0.3333333333333333333333333333333;
        0.101286507323456338800987361915123, 0.101286507323456338800987361915123;
        0.79742698535308732239802527616975, 0.101286507323456338800987361915123;
        0.101286507323456338800987361915123, 0.79742698535308732239802527616975;
        0.470142064105115089770441209513447, 0.470142064105115089770441209513447;
        0.059715871789769820459117580973106, 0.470142064105115089770441209513447;
        0.470142064105115089770441209513447, 0.059715871789769820459117580973106];
    ws = [0.1125; 0.0629695902724135762978419727500906;
        0.0629695902724135762978419727500906;
        0.0629695902724135762978419727500906;
        0.0661970763942530903688246939165759;
        0.0661970763942530903688246939165759;
        0.0661970763942530903688246939165759];
elseif degree <= 6 %degree 6
    nqp = int32(12);
    cs = [0.063089014491502228340331602870819, 0.063089014491502228340331602870819;
        0.063089014491502228340331602870819, 0.87382197101699554331933679425836;
        0.87382197101699554331933679425836, 0.063089014491502228340331602870819;
        0.24928674517091042129163855310702, 0.24928674517091042129163855310702;
        0.24928674517091042129163855310702, 0.50142650965817915741672289378596;
        0.50142650965817915741672289378596, 0.24928674517091042129163855310702;
        0.053145049844816947353249671631398, 0.31035245103378440541660773395655;
        0.31035245103378440541660773395655, 0.053145049844816947353249671631398;
        0.053145049844816947353249671631398, 0.63650249912139864723014259441205;
        0.63650249912139864723014259441205, 0.053145049844816947353249671631398;
        0.31035245103378440541660773395655, 0.63650249912139864723014259441205;
        0.63650249912139864723014259441205, 0.31035245103378440541660773395655];
    ws = [0.025422453185103408460468404553434;
        0.025422453185103408460468404553434;
        0.025422453185103408460468404553434;
        0.05839313786318968301264480569279;
        0.05839313786318968301264480569279;
        0.05839313786318968301264480569279;
        0.041425537809186787596776728210221;
        0.041425537809186787596776728210221;
        0.041425537809186787596776728210221;
        0.041425537809186787596776728210221;
        0.041425537809186787596776728210221;
        0.041425537809186787596776728210221];
elseif degree <= 7 %degree 6&7
    nqp = int32(12);
    cs = [0.062382265094402118, 0.067517867073916085; ...
        0.870099867831681797, 0.062382265094402118; ...
        0.067517867073916085, 0.870099867831681797; ...
        0.055225456656926611, 0.32150249385198182; ...
        0.623272049491092, 0.055225456656926611; ...
        0.32150249385198182, 0.623272049491092; ...
        0.034324302945097146, 0.66094919618673565; ...
        0.304726500868167, 0.034324302945097146; ...
        0.66094919618673565, 0.304726500868167; ...
        0.51584233435359177, 0.27771616697639178; ...
        0.206441498670016, 0.515842334353592; ...
        0.27771616697639178, 0.206441498670016];
    ws = [0.026517028157436251; ...
        0.026517028157436251; ...
        0.026517028157436251; ...
        0.043881408714446055; ...
        0.043881408714446055; ...
        0.043881408714446055; ...
        0.028775042784981585; ...
        0.028775042784981585; ...
        0.028775042784981585; ...
        0.067493187009802774; ...
        0.067493187009802774; ...
        0.067493187009802774];
elseif degree <= 8
    nqp = int32(16);
    cs = [0.33333333333333333333333333333333, 0.33333333333333333333333333333333;
        0.1705693077517602066222935014994, 0.1705693077517602066222935014994;
        0.1705693077517602066222935014994, 0.65886138449647958675541299700121;
        0.65886138449647958675541299700121, 0.1705693077517602066222935014994;
        0.050547228317030975458423550596387, 0.050547228317030975458423550596387;
        0.050547228317030975458423550596387, 0.89890554336593804908315289880723;
        0.89890554336593804908315289880723, 0.050547228317030975458423550596387;
        0.45929258829272315602881551450124, 0.45929258829272315602881551450124;
        0.45929258829272315602881551450124, 0.081414823414553687942368970997513;
        0.081414823414553687942368970997513, 0.45929258829272315602881551450124;
        0.72849239295540428124100037918962, 0.26311282963463811342178578626121;
        0.26311282963463811342178578626121, 0.72849239295540428124100037918962;
        0.72849239295540428124100037918962, 0.0083947774099576053372138345491687;
        0.0083947774099576053372138345491687, 0.72849239295540428124100037918962;
        0.26311282963463811342178578626121, 0.0083947774099576053372138345491687;
        0.0083947774099576053372138345491687, 0.26311282963463811342178578626121;
        ];
    ws = [0.072157803838893584125545555249701;
        0.051608685267359125140895775145648;
        0.051608685267359125140895775145648;
        0.051608685267359125140895775145648;
        0.016229248811599040155462964170437;
        0.016229248811599040155462964170437;
        0.016229248811599040155462964170437;
        0.047545817133642312396948052190887;
        0.047545817133642312396948052190887;
        0.047545817133642312396948052190887;
        0.013615157087217497132422345038231;
        0.013615157087217497132422345038231;
        0.013615157087217497132422345038231;
        0.013615157087217497132422345038231;
        0.013615157087217497132422345038231;
        0.013615157087217497132422345038231; ...
        ];
elseif degree <= 9 %degree 9
    nqp = int32(19);
    cs = [0.33333333333333333, 0.33333333333333333; ...
        0.48968251919873762, 0.48968251919873762; ...
        0.48968251919873762, 0.02063496160252476; ...
        0.02063496160252476, 0.48968251919873762; ...
        0.43708959149293663, 0.43708959149293663; ...
        0.43708959149293663, 0.125820817014127; ...
        0.125820817014127, 0.43708959149293663; ...
        0.18820353561903273, 0.18820353561903273; ...
        0.62359292876193454, 0.18820353561903273; ...
        0.18820353561903273, 0.62359292876193454; ...
        0.044729513394452709, 0.044729513394452709; ...
        0.910540973211094582, 0.044729513394452709; ...
        0.044729513394452709, 0.910540973211094582; ...
        0.74119859878449802, 0.036838412054736283; ...
        0.036838412054736283, 0.74119859878449802; ...
        0.74119859878449802, 0.221962989160765697; ...
        0.036838412054736283, 0.221962989160765697; ...
        0.221962989160765697, 0.036838412054736283; ...
        0.221962989160765697, 0.74119859878449802; ...
        ];
    ws = [0.048567898141399416; ...
        0.015667350113569535; ...
        0.015667350113569535; ...
        0.015667350113569535; ...
        0.038913770502387139; ...
        0.038913770502387139; ...
        0.038913770502387139; ...
        0.039823869463605126; ...
        0.039823869463605126; ...
        0.039823869463605126; ...
        0.012788837829349015; ...
        0.012788837829349015; ...
        0.012788837829349015; ...
        0.021641769688644688; ...
        0.021641769688644688; ...
        0.021641769688644688; ...
        0.021641769688644688; ...
        0.021641769688644688; ...
        0.021641769688644688; ...
        ];
elseif degree <= 10 %degree 10
    nqp = int32(25);
    cs = [0.33333333333333333333333333333333, 0.33333333333333333333333333333333;
        0.42508621060209057296952951163804, 0.42508621060209057296952951163804;
        0.42508621060209057296952951163804, 0.14982757879581885406094097672391;
        0.14982757879581885406094097672391, 0.42508621060209057296952951163804;
        0.02330886751000019071446638689598, 0.02330886751000019071446638689598;
        0.02330886751000019071446638689598, 0.95338226497999961857106722620804;
        0.95338226497999961857106722620804, 0.02330886751000019071446638689598;
        0.62830740021349255642083766607883, 0.2237669735769730062256864902682;
        0.2237669735769730062256864902682, 0.62830740021349255642083766607883;
        0.62830740021349255642083766607883, 0.14792562620953443735347584365296;
        0.14792562620953443735347584365296, 0.62830740021349255642083766607883;
        0.2237669735769730062256864902682, 0.14792562620953443735347584365296;
        0.14792562620953443735347584365296, 0.2237669735769730062256864902682;
        0.6113138261813976489187550022539, 0.35874014186443146457815530072385;
        0.35874014186443146457815530072385, 0.6113138261813976489187550022539;
        0.6113138261813976489187550022539, 0.029946031954170886503089697022247;
        0.029946031954170886503089697022247, 0.6113138261813976489187550022539;
        0.35874014186443146457815530072385, 0.029946031954170886503089697022247;
        0.029946031954170886503089697022247, 0.35874014186443146457815530072385;
        0.82107206998562937337354441347218, 0.14329537042686714530585663061732;
        0.14329537042686714530585663061732, 0.82107206998562937337354441347218;
        0.82107206998562937337354441347218, 0.0356325595875034813205989559105;
        0.0356325595875034813205989559105, 0.82107206998562937337354441347218;
        0.14329537042686714530585663061732, 0.0356325595875034813205989559105;
        0.0356325595875034813205989559105, 0.14329537042686714530585663061732];
    
    ws = [0.039947252370619853915623522606693;
        0.035561901116188667319645643699329;
        0.035561901116188667319645643699329;
        0.035561901116188667319645643699329;
        0.0041119093452320977593233101812359;
        0.0041119093452320977593233101812359;
        0.0041119093452320977593233101812359;
        0.022715296148085009003536814621966;
        0.022715296148085009003536814621966;
        0.022715296148085009003536814621966;
        0.022715296148085009003536814621966;
        0.022715296148085009003536814621966;
        0.022715296148085009003536814621966;
        0.018679928117152638413118249500988;
        0.018679928117152638413118249500988;
        0.018679928117152638413118249500988;
        0.018679928117152638413118249500988;
        0.018679928117152638413118249500988;
        0.018679928117152638413118249500988;
        0.015443328442281994391256538502314;
        0.015443328442281994391256538502314;
        0.015443328442281994391256538502314;
        0.015443328442281994391256538502314;
        0.015443328442281994391256538502314;
        0.015443328442281994391256538502314];
elseif degree <= 12 %degree 12
    % D.A. Dunavant, High degree efficient symmetrical Gaussian quadrature
    % rules for the triangle, Internat. J. Numer. Methods Engrg. 21 (1985), 1129--1148.
    nqp = int32(33);
    cs = [0.48821738977380488256466173878598, 0.48821738977380488256466173878598;
        0.48821738977380488256466173878598, 0.023565220452390234870676522428033;
        0.023565220452390234870676522428033, 0.48821738977380488256466173878598;
        0.43972439229446027297973620450348, 0.43972439229446027297973620450348;
        0.43972439229446027297973620450348, 0.12055121541107945404052759099305;
        0.12055121541107945404052759099305, 0.43972439229446027297973620450348;
        0.27121038501211592234595160781199, 0.27121038501211592234595160781199;
        0.27121038501211592234595160781199, 0.45757922997576815530809678437601;
        0.45757922997576815530809678437601, 0.27121038501211592234595160781199;
        0.12757614554158592467389281696323, 0.12757614554158592467389281696323;
        0.12757614554158592467389281696323, 0.74484770891682815065221436607355;
        0.74484770891682815065221436607355, 0.12757614554158592467389281696323;
        0.021317350453210370246857737134961, 0.021317350453210370246857737134961;
        0.021317350453210370246857737134961, 0.95736529909357925950628452573008;
        0.95736529909357925950628452573008, 0.021317350453210370246857737134961;
        0.11534349453469799916901160654623, 0.2757132696855141939747907691782;
        0.2757132696855141939747907691782, 0.11534349453469799916901160654623;
        0.11534349453469799916901160654623, 0.60894323577978780685619762427557;
        0.60894323577978780685619762427557, 0.11534349453469799916901160654623;
        0.2757132696855141939747907691782, 0.60894323577978780685619762427557;
        0.60894323577978780685619762427557, 0.2757132696855141939747907691782;
        0.022838332222257029610233386418649, 0.28132558098993954824813282149259;
        0.28132558098993954824813282149259, 0.022838332222257029610233386418649;
        0.022838332222257029610233386418649, 0.69583608678780342214163379208876;
        0.69583608678780342214163379208876, 0.022838332222257029610233386418649;
        0.28132558098993954824813282149259, 0.69583608678780342214163379208876;
        0.69583608678780342214163379208876, 0.28132558098993954824813282149259;
        0.025734050548330228168108745174704, 0.11625191590759714124135593566697;
        0.11625191590759714124135593566697, 0.025734050548330228168108745174704;
        0.025734050548330228168108745174704, 0.85801403354407263059053531915832;
        0.85801403354407263059053531915832, 0.025734050548330228168108745174704;
        0.11625191590759714124135593566697, 0.85801403354407263059053531915832;
        0.85801403354407263059053531915832, 0.11625191590759714124135593566697];
    
    ws = [0.012865533220227667708895587247731;
        0.012865533220227667708895587247731;
        0.012865533220227667708895587247731;
        0.021846272269019201067729355264938;
        0.021846272269019201067729355264938;
        0.021846272269019201067729355264938;
        0.031429112108942550177134995670765;
        0.031429112108942550177134995670765;
        0.031429112108942550177134995670765;
        0.017398056465354471494663093004469;
        0.017398056465354471494663093004469;
        0.017398056465354471494663093004469;
        0.0030831305257795086169334151704928;
        0.0030831305257795086169334151704928;
        0.0030831305257795086169334151704928;
        0.020185778883190464758914841227262;
        0.020185778883190464758914841227262;
        0.020185778883190464758914841227262;
        0.020185778883190464758914841227262;
        0.020185778883190464758914841227262;
        0.020185778883190464758914841227262;
        0.011178386601151722855919352997536;
        0.011178386601151722855919352997536;
        0.011178386601151722855919352997536;
        0.011178386601151722855919352997536;
        0.011178386601151722855919352997536;
        0.011178386601151722855919352997536;
        0.0086581155543294461858209159291448;
        0.0086581155543294461858209159291448;
        0.0086581155543294461858209159291448;
        0.0086581155543294461858209159291448;
        0.0086581155543294461858209159291448;
        0.0086581155543294461858209159291448];
else %degree 13
    nqp = int32(37);
    cs = [0.33333333333333333333333333333333, 0.33333333333333333333333333333333;
        0.49504818493970466551410613458718, 0.49504818493970466551410613458718;
        0.49504818493970466551410613458718, 0.0099036301205906689717877308256304;
        0.0099036301205906689717877308256304, 0.49504818493970466551410613458718;
        0.46871663510957383858305759615608, 0.46871663510957383858305759615608;
        0.46871663510957383858305759615608, 0.062566729780852322833884807687836;
        0.062566729780852322833884807687836, 0.46871663510957383858305759615608;
        0.41452133680127650292770411894038, 0.41452133680127650292770411894038;
        0.41452133680127650292770411894038, 0.17095732639744699414459176211924;
        0.17095732639744699414459176211924, 0.41452133680127650292770411894038;
        0.22939957204283144126863890407517, 0.22939957204283144126863890407517;
        0.22939957204283144126863890407517, 0.54120085591433711746272219184966;
        0.54120085591433711746272219184966, 0.22939957204283144126863890407517;
        0.11442449519632999965001086032284, 0.11442449519632999965001086032284;
        0.11442449519632999965001086032284, 0.77115100960734000069997827935433;
        0.77115100960734000069997827935433, 0.11442449519632999965001086032284;
        0.024811391363458980461148451658982, 0.024811391363458980461148451658982;
        0.024811391363458980461148451658982, 0.95037721727308203907770309668204;
        0.95037721727308203907770309668204, 0.024811391363458980461148451658982;
        0.094853828379578994378640288687166, 0.26879499705876077881194860407488;
        0.26879499705876077881194860407488, 0.094853828379578994378640288687166;
        0.094853828379578994378640288687166, 0.63635117456166022680941110723796;
        0.63635117456166022680941110723796, 0.094853828379578994378640288687166;
        0.26879499705876077881194860407488, 0.63635117456166022680941110723796;
        0.63635117456166022680941110723796, 0.26879499705876077881194860407488;
        0.01810077327880705425651607573572, 0.29173006673428778551596130342026;
        0.29173006673428778551596130342026, 0.01810077327880705425651607573572;
        0.01810077327880705425651607573572, 0.69016915998690516022752262084402;
        0.69016915998690516022752262084402, 0.01810077327880705425651607573572;
        0.29173006673428778551596130342026, 0.69016915998690516022752262084402;
        0.69016915998690516022752262084402, 0.29173006673428778551596130342026;
        0.022233076674090070566811992949056, 0.12635738549166876785161724506539;
        0.12635738549166876785161724506539, 0.022233076674090070566811992949056;
        0.022233076674090070566811992949056, 0.85140953783424116158157076198556;
        0.85140953783424116158157076198556, 0.022233076674090070566811992949056;
        0.12635738549166876785161724506539, 0.85140953783424116158157076198556;
        0.85140953783424116158157076198556, 0.12635738549166876785161724506539];
    
    ws = [0.026260461700400838620392011180488;
        0.0056400726046647664395273501759664;
        0.0056400726046647664395273501759664;
        0.0056400726046647664395273501759664;
        0.01571175918122722544615733095307;
        0.01571175918122722544615733095307;
        0.01571175918122722544615733095307;
        0.023536251252097113664615898114823;
        0.023536251252097113664615898114823;
        0.023536251252097113664615898114823;
        0.023681793268177187316493777028326;
        0.023681793268177187316493777028326;
        0.023681793268177187316493777028326;
        0.015583764522896899256045459475966;
        0.015583764522896899256045459475966;
        0.015583764522896899256045459475966;
        0.0039878857325371862231920366094715;
        0.0039878857325371862231920366094715;
        0.0039878857325371862231920366094715;
        0.018424201364366132610170069259958;
        0.018424201364366132610170069259958;
        0.018424201364366132610170069259958;
        0.018424201364366132610170069259958;
        0.018424201364366132610170069259958;
        0.018424201364366132610170069259958;
        0.0087007316519110832524098255599759;
        0.0087007316519110832524098255599759;
        0.0087007316519110832524098255599759;
        0.0087007316519110832524098255599759;
        0.0087007316519110832524098255599759;
        0.0087007316519110832524098255599759;
        0.0077608934195224551943388438046503;
        0.0077608934195224551943388438046503;
        0.0077608934195224551943388438046503;
        0.0077608934195224551943388438046503;
        0.0077608934195224551943388438046503;
        0.0077608934195224551943388438046503];
end

end
