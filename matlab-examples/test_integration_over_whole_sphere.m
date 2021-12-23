function test_integration_over_whole_sphere()
    % This is a simple demo to show how we use ARPIST.
    % You can directly run this script.
    % To be noted that, the three test meshes we provided are extremely coarse to avoiding uploading large files into github.

    %% add path for this demo
    addpath('../matlab-implementation');
    addpath('../meshes');
    %% parameters
    % test degrees
    degs = [-1, 2, 4, 6, 8];
    degree_name = {'adaptive ARPIST', 'degree-2 ARPIST', 'degree-4 ARPIST', ...
                    'degree-6 ARPIST', 'degree-8 ARPIST'};
    num_deg = length(degs);
    % color and line styles for different degrees
    c = {'-or', '-+k', '--sb', '-.^m', '-*g'};

    % test functions
    function_handles = {@f1, @f3, @sin_cos_exp};
    num_fun = length(function_handles);
    % exact spherical integration for each function
    exact_integrations = [4 * pi / 9, 0.049629692928687, 4 * pi / exp(1)];

    % test meshs
    mesh_names = {'SphereMesh_N=64_r=1.mat', 'SphereMesh_N=256_r=2.mat', ...
            'SphereMesh_N=1024_r=3000.mat'};
    num_mesh = length(mesh_names);
    % radius for different meshes
    r = zeros(1, 3);
    % number of elements for each mesh
    num_elems = zeros(1, 3);

    % integration errors
    errs = zeros(num_deg, num_mesh, num_fun);

    %% Spherical Integration
    for mesh_id = 1:length(mesh_names)
        mesh_mat = load(mesh_names{mesh_id});
        r(mesh_id) = norm(mesh_mat.xs(1, :));
        num_elems(mesh_id) = size(mesh_mat.surfs, 1);
        fprintf("====================================\n");
        fprintf("Get into mesh %d\n", mesh_id);

        for ii = 1:length(degs)
            deg = degs(ii);
            %% Compute cell area by integrating constant 1
            [areas] = spherical_integration(mesh_mat.xs, mesh_mat.surfs, ...
            @(x)(1), deg);
            sum_area = sum(areas);
            fprintf("Area of sphere for %s: %e\n", degree_name{ii}, sum_area);
            exact_area = 4 * r(mesh_id)^2 * pi;
            fprintf("Relative error: %e\n", abs(sum_area - exact_area) / exact_area);

            %% compute integration of function using ARPIST
            if (deg > 0)
                % if degree is set up, use the given degree
                [pnts, ws, ptr] = compute_sphere_quadrature(mesh_mat.xs, mesh_mat.surfs, 100, deg);
            else
                % if degree is not set up, use our configuration for adaptive ARPIST
                [pnts, ws, ptr] = compute_sphere_quadrature(mesh_mat.xs, mesh_mat.surfs);
            end

            % number of quadrature points
            nqp = size(pnts, 1);
            % function value at each quadrature points
            fs_qp = zeros(nqp, num_fun);

            for function_id = 1:num_fun
                fun_handle = function_handles{function_id};
                % Another simple implementation if we only have one test function.
                % Just like what we did for area integration.
                [fs] = spherical_integration(mesh_mat.xs, mesh_mat.surfs, ...
                fun_handle, deg);

                for qpid = 1:nqp
                    fs_qp(qpid, function_id) = fun_handle(pnts(qpid, :));
                end

            end

            % numerical integrations
            nints = ws' * fs_qp;

            for function_id = 1:num_fun
                exact_int = exact_integrations(function_id) * r(mesh_id)^2;
                errs(ii, mesh_id, function_id) = abs(nints(function_id) - ...
                    exact_int) / exact_int;
                % print out relative error
                fprintf("Relative error for function %d with %s: %e\n", ...
                function_id, degree_name{ii}, errs(ii, mesh_id, function_id));
            end

        end

    end

    %% compare errors for different ARPIST
    font_size = 12;

    for function_id = 1:num_fun
        figure('DefaultAxesFontSize', font_size);

        for ii = 1:num_deg
            loglog(num_elems, errs(ii, :, function_id), c{ii});
            hold on;
        end

        xlabel('Number of elements');
        ylabel('Integration errors');
        ylim([1e-17, 1e7]);
        legend(degree_name, 'Location', 'NorthEast', 'FontSize', font_size);
    end

end

function y = f1(x)
    x = x / norm(x);
    y = (1 + tanh((x(3) - x(1) - x(2)) * 9)) / 9;
end

function y = f3(x)
    x = x / norm(x);
    y = 0.5 + atan(300 * (x(3) - 0.9999)) / pi;
end

function y = sin_cos_exp(x)
    x = x / norm(x);
    y = exp(x(1)) * (x(2)^2 + x(1) * sin(x(2))) + x(2) * cos(x(3));
end

function fs = evaluate_f(xs, f_D)
    nv = size(xs, 1);
    fs = zeros(nv, 1);

    for vid = 1:nv
        fs(vid) = f_D(xs(vid, :));
    end

end
