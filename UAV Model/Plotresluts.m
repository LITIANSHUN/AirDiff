%% Parameter Initialization
Tsc               = 1e-3;
Ts                = Tsc;

infPlane.x        = 12.5;
infPlane.y        = 8.5;
infPlane.z        = 0.2;

pkgSize           = [0.15 0.15 0.15];
pkgDensity        = 5;

pkgGrndStiff      = 10000;
pkgGrndDamp       = 30;
pkgGrndTransW     = 1e-6;

propeller.diameter    = 0.254;
propeller.hover_speed = 700;

x0 = -5;
y0 = -3;
z0 = 0.06;

xrot = 0;
yrot = 0;
zrot = 0;

rho_nylon = 1.41;
rho_glass = 2.56;
rho_pla   = 1.25;
rho_cfrp  = 2.1067;
rho_al    = 2.66;

x_waypt = [-5 -2.5 0 2.5 5]';
y_waypt = [-3 -1.5 0 1.5 3]';
z_waypt = [6 6 6 6 1]';
waypoints = [x_waypt y_waypt z_waypt]';

xTrajPts = [x0; x_waypt];
yTrajPts = [y0; y_waypt];
zTrajPts = [z0; z_waypt];

V_nominal = 1;

timespot = zeros(numel(x_waypt),1);
for i = 1:numel(x_waypt)
    timespot(i) = max(abs([xTrajPts(i)-xTrajPts(i+1), ...
                           yTrajPts(i)-yTrajPts(i+1), ...
                           zTrajPts(i)-zTrajPts(i+1)]))/V_nominal;
end

T_stop   = 25;
targetX  = waypoints(1,end);
targetY  = waypoints(2,end);
targetZ  = waypoints(3,end);

kp_position = 0.0175;
kd_position = 0.85;

kp_attitude = 8.5;
ki_attitude = 5;
kd_attitude = 40;

kp_altitude = 0.15;
ki_altitude = 0.0275;
kd_altitude = 0.475;

%% Plotting
if ~exist('simlogQuadcopter','var')
    sim('Quadcopter');
end

if ~exist('h1_Quadcopter','var') || ~isgraphics(h1_Quadcopter,'figure')
    h1_Quadcopter = figure('Name','Quadcopter');
end
figure(h1_Quadcopter); clf(h1_Quadcopter);

xpos = logsoutQuadcopter.get('xpos');
ypos = logsoutQuadcopter.get('ypos');
zpos = logsoutQuadcopter.get('zpos');
xpos_des = logsoutQuadcopter.get('xpos_des');
ypos_des = logsoutQuadcopter.get('ypos_des');
zpos_des = logsoutQuadcopter.get('zpos_des');

subplot(3,1,1);
plot(xpos_des.Values.Time, xpos_des.Values.Data,'LineWidth',1); hold on;
plot(xpos.Values.Time,     xpos.Values.Data,    'LineWidth',1);
grid on; title('Position of the Quadcopter');
ylabel('X Position (m)'); xlabel('Time (s)');
axis([0 25 -8 8]);
legend('Desired','Actual','Location','best').Box = 'off';

subplot(3,1,2);
plot(ypos_des.Values.Time, ypos_des.Values.Data,'LineWidth',1); hold on;
plot(ypos.Values.Time,     ypos.Values.Data,    'LineWidth',1);
grid on; ylabel('Y Position (m)'); xlabel('Time (s)');
axis([0 25 -5 5]);

subplot(3,1,3);
plot(zpos_des.Values.Time, zpos_des.Values.Data,'LineWidth',1); hold on;
plot(zpos.Values.Time,     zpos.Values.Data,    'LineWidth',1);
grid on; ylabel('Z Position (m)'); xlabel('Time (s)');
axis([0 25 0 8]);

linkaxes(findobj(h1_Quadcopter,'Type','axes'),'x');
clear xpos ypos zpos xpos_des ypos_des zpos_des;