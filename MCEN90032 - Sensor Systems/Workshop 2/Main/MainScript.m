clear all;
fname = 'step1';
file = load('step1.mat');

% Inital heading (real world coordinate)
init_compass_heading = 35;
% Rotation matrix to re-map compass coordinate to match world coordinates
r = roty(180)*rotz(90);

% Hard-define the plotting lim based on my Google Maps screenshot
img = imread("map.png");
xl = [-7,37];
yl = [35,-19];
%% Ground truth plotting


fig=gcf;
fig.Position(4)=653;

% Google map scaling ratio is used as frame of reference 
% to estimate checkpoint coordinates
% 3 checkpoints
cp1 = [-37.799331, 144.961263, 0];
cp2 = [-37.79916028212808, 144.9616082566491, 0];
cp3 = [-37.799342, 144.961525, 0];
% or -37.799360, 144.961542

% Get movement vector between 3 checkpoints
r12 = lla2enu(cp2,cp1,'flat');
r23 = lla2enu(cp3,cp2,'flat');
r31 = lla2enu(cp1,cp3,'flat');

% Plotting
checkpoints_x = [0,r12(1),r12(1)+r23(1),0] ;
checkpoints_y = [0,r12(2),r12(2)+r23(2),0] ;
true_norm = [0,norm(r12),norm(r12)+norm(r23),norm(r12)+norm(r23)+norm(r31)];
plot(checkpoints_x,checkpoints_y,'c--o',LineWidth=3)
title('Ground truth, marked on map (Clockwise Run)');
xlabel('x (m)');
ylabel('y (m)');

% Add map for reference
hold on;
uistack(image(xl,yl,img),'bottom');
plot([0],[0],'ro',LineWidth=3);
legend('Walked Path','Starting Point',Location='northeast')
xlim(xl);
ylim([yl(2),yl(1)]);
hold off;
%% GPS plotting


p = file.Position;
loc = [0,0,0];
cps = zeros(size(p,1),3);
for i=1:size(p,1)-1
    % Conversion of lat-long to meter displacement
    r0 = lla2enu([p.latitude(i+1),p.longitude(i+1),0], ...
        [p.latitude(i),p.longitude(i),0],'flat');
    loc = loc + r0;
    cps(i+1,:) = loc;
end
plot(cps(:,1),cps(:,2),'y-.',LineWidth=3)
title('Path & GPS Result');
xlabel('x (m)');
ylabel('y (m)');

% Add map for reference
hold on;
uistack(image(xl,yl,img),'bottom');
plot(checkpoints_x,checkpoints_y,'c--o',LineWidth=3)
plot([0],[0],'ro',LineWidth=3);
legend('True Path','GPS Measurement','Starting Point',Location='northeast')
xlim(xl);
ylim([yl(2),yl(1)]);
hold off;
%% Calibration of magnetometer

calibration_time = [10,20];
m = file.MagneticField;
tm = get_t(m);
% Sampling frequency
fm=1/(tm(2)-tm(1));

% Calibration
calib_idx_start = round(calibration_time(1) * fm);
calib_idx_end = round(calibration_time(2) * fm);
tc = tm(calib_idx_start:calib_idx_end);
xc = m.X(calib_idx_start:calib_idx_end);
yc = m.Y(calib_idx_start:calib_idx_end);
zc = m.Z(calib_idx_start:calib_idx_end);

cm = rad2deg(atan2(yc,xc));
%plot(tc,cm);
[mu_m, sig_m] = normal_dist(cm,ones(calib_idx_end-calib_idx_start+1,1))
%mf_est = unwrap(dm,thresh);
%% Calibration of gyroscope

calibration_time = [10,20];
g = file.AngularVelocity;
tg = get_t(g);
% Sampling frequency
fg=1/(tg(2)-tg(1));

% Calibration
calib_idx_start = round(calibration_time(1) * fm);
calib_idx_end = round(calibration_time(2) * fm);
tc = tg(calib_idx_start:calib_idx_end);
xc = g.X(calib_idx_start:calib_idx_end);
yc = g.Y(calib_idx_start:calib_idx_end);
zc = g.Z(calib_idx_start:calib_idx_end);

cg = rad2deg(atan2(yc,xc));
%plot(tc,cg);
[mu_g, sig_g] = normal_dist(cg,ones(calib_idx_end-calib_idx_start+1,1))
%% Acceleration path info retrieval

fig=gcf;
fig.Position(4)=210;
%%
[ta,a_norm] = get_a(file);
% Get average walking frequency, filtered linear acceleration norm, peak infos
[avg_walking_freq,~,a_fil, p, tp, wp] = step_count(a_norm,ta);
xlabel('Time (s)');
ylabel('Acceleration (m/s^2)');
title('Step Counting on Acceleration Data');
%%
% Find corresponding index of each step in the time array
idx=zeros(length(tp));
for i=1:length(tp)
    [~,idx(i)] = min(abs(ta-tp(i)));
end
% Initial speed estimation using peak magnitude
speed = 1*avg_walking_freq;
% Acummulated speed using definite integral
ps = 2*p.*wp/pi;
% Strides = t*(acummulated speed + initial speed)
strides = ps.*wp + p.*wp;
%%
% Get all stopping periods in the from of [start stop]
min_walk_freq = 1;
% Include 1st and last bit of time, if we were stopping at start or end
tp0 = [0;tp;tg(end)];
dtp = diff(tp0);
ranges = [];
for i = 1:length(dtp)
    if dtp(i) > 1/min_walk_freq
        ranges = [ranges ; [tp0(i) tp0(i+1)]];
    end
end
ranges
%%
% Get distance estimation using double integration
inta = int_est(a_fil,ta);
plot(ta,inta);
title('Distance estimation with Double Integration');
xlabel('Time (s)');
ylabel('Distance (m)');
%% Plot out heading and estimations

[dt,dm,da] = heading(fname,init_compass_heading,ranges);
%% Kalman Filtering

T = 1/fg;
phi = [1  T;
       0  1];
H = [1 0];

Q = [sig_m^2   0;
     0      sig_g^2/2];
R1 = sig_m^2;
R2 = sig_g^2/2;

xe0 = [dm(calib_idx_start),0]';
P00 = zeros(length(phi));%Q;
dx0 = [0,0]';

[kest,kpred,kco] = kf(dm,g.Z,phi,H,Q,R1,R2,xe0,P00,dx0);
plot(tg,[kest(:,2),kpred(:,1),dm,da])
title('Kalman Filter results');
xlabel('Time (s)');
ylabel('Heading (deg)');
legend('Estimation','Prediction','Magnetometer Measurement','Gyro Measurement',Location='best')
%% Plot path estimation, assume constant stride and walking frequency

fig=gcf;
fig.Position(4)=653;
%%
% Hard-defined step length by measurement: Stride*Walk Freq/Sampling Freq
sl = 0.65*avg_walking_freq/fg;

len = length(kest);
coord = zeros(len,2);
coord(1,:) = [0,0];

cse = zeros(len,1);
cse(1) = 0;

for i = 1:len-1
    step=sl;
    for j=1:size(ranges,1)
        if tg(i) < ranges(j,2) && tg(i) > ranges(j,1)
            step = 0;
            break;
        end
    end
    incr = step*r*[cos(deg2rad(kest(i)));sin(deg2rad(kest(i)));0];
    coord(i+1,:) = coord(i,:) + incr([1,2])';
    cse(i+1) = cse(i) + step;
end

plot(coord(:,1),coord(:,2),'c-',LineWidth=3)
title('Step count-based estimation');
xlabel('x (m)');
ylabel('y (m)');

% Add map for reference
hold on;
uistack(image(xl,yl,imread("map.png")),'bottom');
plot(checkpoints_x,checkpoints_y,'--o',LineWidth=3,Color=[194, 232, 63]/255,MarkerEdgeColor='g');
plot([0],[0],'ro',LineWidth=3);
legend('Estimated Path','True Path','Starting Point',Location='northeast')
xlim(xl);
ylim([yl(2),yl(1)]);
hold off;
%% Plot path estimation using peak step acceleration

len = length(idx);
coorx = zeros(len,2);
coorx(1,:) = [0,0];

ase = zeros(len,1);
ase(1) = 0;
for i = 1:len-1
    incr = strides(i)*r*[cos(deg2rad(kest(idx(i))));sin(deg2rad(kest(idx(i))));0];
    coorx(i+1,:) = coorx(i,:) + incr([1,2])';
    ase(i+1) = ase(i) + strides(i);
end

plot(coorx(:,1),coorx(:,2),'c-',LineWidth=3)
title('Acceleration-based estimation');
xlabel('x (m)');
ylabel('y (m)');

% Add map for reference
hold on;
uistack(image(xl,yl,imread("map.png")),'bottom');
plot(checkpoints_x,checkpoints_y,'--o',LineWidth=3,Color=[194, 232, 63]/255,MarkerEdgeColor='g');
plot([0],[0],'ro',LineWidth=3);
legend('Estimated Path','True Path','Starting Point',Location='northeast')
xlim(xl);
ylim([yl(2),yl(1)]);
hold off;
%% Walking distance estimators comparison

plot([0 5.8 28 43 60],[0 true_norm],'-o');
hold on;
plot(tp,ase,'-');
plot(tg,cse,'-');
title('Comparison between walking distance estimations');
xlabel('Time (s)');
ylabel('Distance walked (m)');
legend('Truth','ASE','CSE',Location='best');
hold off;
fig=gcf;
fig.Position(4)=210;
%%
function [est,pred,K] = kf(y,y1,phi,H,Q,R1,R2,xe0,P00,dx0)
    l = length(y);
    sz = length(xe0);
    xek1k1 = xe0;
    est = zeros(l,2);
    pred = zeros(l,sz);

    dx = dx0;

    Pkk = P00;

    for k=1:l-1
        % Prediction
        xekk = xek1k1;
        xek1k = phi*xekk;
        Pk1k = phi*Pkk*phi' + Q;
        pred(k,:) = xek1k;
        % Update
        K = Pk1k*H' / (H*Pk1k*H'+R1);
        xek1k1 = xek1k + K*(y(k+1)-H*xek1k);
        I = eye(length(H));
        k_1 = I-K*H;
        Pkk = k_1*Pk1k*k_1' + K*R1*K';
        est(k,1) = xek1k1(1);
        K = Pk1k*H' / (H*Pk1k*H'+R2);
        xek1k1 = xek1k1 + K*(y1(k+1)-H*xek1k1);
        I = eye(length(H));
        k_1 = I-K*H;
        Pkk = k_1*Pkk*k_1' + K*R2*K';
        % Estimation
        est(k,2) = xek1k1(1);
    end
end

function [truth,mf_est,av_est] = heading(filename,compass,srange)
    data = load(filename+".mat");
    m = data.MagneticField;
    a = data.AngularVelocity;
    o = data.Orientation;
    % a_data = [a.X, a.Y, a.Z];
    m_data = [m.X, m.Y, m.Z];
    % o_data = [o.X, o.Y, o.Z];
    
    ta = get_t(a);
    tm = get_t(m);
    to = get_t(o);
    % Sampling frequency
    f=1/(ta(2)-ta(1));
    
    % Discontinuity threshold in degrees to unwrap
    thresh = 300;
    
    % True orientation. Phone is facing up, so only X changes
    truth = unwrap(o.X,thresh);
    
    %% Magnetometer calibration
    % Iron effects removal
    [A,b,~]  = magcal(m_data);
    M = (m_data-b)*A;
    
    %% Heading estimation using magnetic field (Only atan2(y,x) changes)
    dm = rad2deg(atan2(M(:,2),M(:,1)));
    mf_est = unwrap(dm,thresh);
    % Align with compass data to ensure accuracy
    calibration_time = [2,4];
    fm=1/(tm(2)-tm(1));
    calib_idx_start = round(calibration_time(1) * fm);
    calib_idx_end = round(calibration_time(2) * fm);
    % Use linear regression to get the bias values
    tc = tm(calib_idx_start:calib_idx_end);
    mc = mf_est(calib_idx_start:calib_idx_end);
    T = [ones(length(tc),1) tc];
    m_val = T\mc;
    mf_est = mf_est + compass - m_val(1);

    % Remove constant to compare with orientation
    truth = truth + mf_est(1) - truth(1);
    
    % Heading estimation using angular velocity, with tilt correction (Only Z changes)
    da = zeros(length(a.Z),1);
    for i = 1:length(a.Z)-1
        step=1;
        for j=1:size(srange,1)
            if ta(i) < srange(j,2) && ta(i) > srange(j,1)
                step = 0;
                break;
            end
        end
        incr = step*rad2deg(-a.Z(i)/f);
        da(i+1) = da(i) + incr;
    end
    av_est = da+mf_est(1);

    % Low-pass filtering to get rid of noise
    mf_est = lowpass(mf_est,1/2/f,f);
    av_est = lowpass(av_est,1/2/f,f);
    truth = lowpass(truth,1/2/f,f);

    
    % Comparison plot
    plot(to,truth);
    hold on;
    plot(tm,mf_est);
    plot(ta,av_est);
    legend('Orientation meter measure','Magnetic Field estimation','Angular Velocity estimation','location','best');
    title(append(filename,' - Comparison between:\newline True Heading and Estimation Methods\newline'));
    xlabel('t (seconds)');
    ylabel('degrees');
    print(append(filename,'_estimation_comparison.png'),'-dpng');
    hold off;
end

function t = get_t(a)
    out = [0;seconds(diff(a.Timestamp))];
    t = cumsum(out);
end

function [t,a_norm] = get_a(file)
    a = file.Acceleration;
    calibration_time = 3;
    
    % Is the phone position stationary?
    stationary = true;
    
    % In case phone position is stationary, set up the gravity
    gravity_x = 0;
    gravity_y = 0;
    gravity_z = 9.8;
    gravity = sqrt(gravity_x^2 + gravity_y^2 + gravity_z^2);
    
    t = get_t(a);
    % Sampling frequency
    f=1/(t(2)-t(1));
    
    % Calibration
    calib_idx = round(calibration_time * f);
    tc = t(1:calib_idx);
    xc = a.X(1:calib_idx);
    yc = a.Y(1:calib_idx);
    zc = a.Z(1:calib_idx);
    
    if stationary
        % Use linear regression to get the bias values
        T = [ones(length(tc),1) tc];
        line_x = T\xc;
        X0 = line_x(1) - gravity_x;
        line_y = T\yc;
        Y0 = line_y(1) - gravity_y;
        line_z = T\zc;
        Z0 = line_z(1) - gravity_z;
    else
        [X0, Y0, Z0] = deal(0,0,0);
    end
    % Norm by time
    a_norm = ((a.X-X0).^2 + (a.Y-Y0).^2 + (a.Z-Z0).^2).^0.5;
end

function dv = int_est(a_norm,t)
    da = cumtrapz(t,a_norm-9.8);
    dv = cumtrapz(t,da);
end

function [mu,sig] = normal_dist(x,p)
    sum_x = 0;
    sum_var = 0;

    sum_p = sum(p);
    for i=1:length(x)
        sum_x = sum_x + x(i)*p(i)/sum_p;
    end
    mu = sum_x;
    for i=1:length(x)
        sum_var = sum_var + (x(i)-mu)^2;
    end
    sig = sqrt(sum_var/sum_p);
end

function [mu,sig,a_filtered,ps,lps,wp] = step_count(a_norm,t)
    % Is the phone position stationary?
    stationary = true;
    gravity = 9.8;
    f=1/(t(2)-t(1));
    calibration_time = 3;
    calib_idx = round(calibration_time * f);

    a_tf = fft(a_norm - gravity);
    a_tf = fftshift(a_tf);
    L = length(a_norm);
    freqs = f/L * linspace(0,L-1,L) - f/2;
    
    [pks,loc] = findpeaks(abs(a_tf), freqs);
    l = ceil(length(loc)/2);
    [mu,sig] = normal_dist(loc(l:end),pks(l:end));
    
    % Set up filter
    fmin = mu - 6.5*sig;
    fmax = mu + 3.5*sig;
    % Filter
    a_filtered = bandpass(a_norm,[fmin,fmax],f);
    
    % Use calibration segment to identify max prominence noise
    [~,~,~,pc] = findpeaks(a_filtered(1:calib_idx), freqs(1:calib_idx), 'SortStr','descend');
    
    % Ensure margin is large enough to prevent noise peaks
    b_norm = 0;%(X0^2+Y0^2+Z0^2)^0.5;
    if b_norm > 0.2 margin=b_norm; else margin=0.2; end 
    
    minP = pc(1) + margin; % Add bias norm as safe margin
    
    % Get the highest likely step frequency in the data
    maxStepFreq = mu + 4*sig;
    
    % Set up peak-finding parameters
    settings = struct('MinPeakDistance', 1/maxStepFreq, ...
                      'MinPeakProminence', minP);
    
    % Find peaks that correspond to steps
    [ps,lps,wp,~] = findpeaks(a_filtered,t,settings);
    findpeaks(a_filtered,t,settings);
end

function r = roty(t)
    dt = deg2rad(t);
    r = [cos(dt) 0 sin(dt);
        0        1       0;
        -sin(dt) 0 cos(dt)];
end

function r = rotz(t)
    dt = deg2rad(t);
    r = [cos(dt) -sin(dt) 0;
        sin(dt)   cos(dt) 0;
        0         0       1];
end