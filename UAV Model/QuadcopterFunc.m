open_system('Quadcopter');
set_param('Quadcopter','SimMechanicsOpenEditorOnUpdate','off');
set_param(find_system('Quadcopter','FindAll','on','type','annotation','Tag','ModelFeatures'),'Interpreter','off');

set_param('Quadcopter/Package Delivery Quadcopter','LinkStatus','none');
open_system('Quadcopter/Package Delivery Quadcopter','force');

set_param('Quadcopter/Trajectory Generation and Control','LinkStatus','none');
open_system('Quadcopter/Trajectory Generation and Control','force');

set_param('Quadcopter/Trajectory Generation and Control/Position and Attitude Controller','LinkStatus','none');
open_system('Quadcopter/Trajectory Generation and Control/Position and Attitude Controller','force');

PlotResults;
set_param('Quadcopter','SimMechanicsOpenEditorOnUpdate','on');

close all;
bdclose all;
clear;