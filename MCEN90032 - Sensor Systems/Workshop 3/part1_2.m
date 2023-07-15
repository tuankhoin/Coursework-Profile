% data = load('lidarData.mat');
% lbl = load('lidarLabel.mat');
%% Part 1: Create PointCloud

[X,Y] = meshgrid(linspace(-0.1,1.1,25),linspace(-0.1,2.1,40));
Z = (X>=0 & X<=1 & Y>=0 & Y<=2).*(X.^2+Y.^3);
% surf(X,Y,Z)
%%
pc = pointCloud([X(:) Y(:) Z(:)])
pcshow(pc)
title('Formed point cloud');
%% Part 2.1: Transformation

tform = rigid3d(rotz(60),[2,4,0]);
pc_tf = pctransform(pc,tform);
pcshow(pc_tf);
title('Transformed point cloud (60^o rotation, <2,4,0> translation)');
%% Part 2.2-2.3: ICP

% True transform
true_transform = tform.T
% ICP estimation of transform matrix
[tf,~,~] = pcregistericp(pc,pc_tf,MaxIterations=100,Tolerance=[0.01,0.01],Verbose=true);
predicted_transform = tf.T
%%
% Plot out iteration error
e = zeros(26,1);
for i=1:26
    [~,~,e(i)] = pcregistericp(pc,pc_tf,MaxIterations=i,Tolerance=[0.01,0.05]);
end
plot(e,'-o');
title('Error Across Iterations');
xlabel('Number of iterations');
ylabel('Root Mean Squared Error')
%%
function r = roty(t)
    dt = deg2rad(t);
    r = [cos(dt) 0 sin(dt);
        0        1       0;
        -sin(dt) 0 cos(dt)];
end
function r = rotx(t)
    dt = deg2rad(t);
    r = [1  0        0;
         0  cos(dt) -sin(dt);
         0  sin(dt)  cos(dt)];
end
function r = rotz(t)
    dt = deg2rad(t);
    r = [cos(dt) -sin(dt) 0;
        sin(dt)   cos(dt) 0;
        0         0       1];
end