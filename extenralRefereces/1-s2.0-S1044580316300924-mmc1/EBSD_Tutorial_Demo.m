%------------------------------------------------------------------------------
% Simple demonstration/simulation of lattice plane traces,
% stereographic projection and unit cell creating
%------------------------------------------------------------------------------
%
%
%------------------------------------------------------------------------------
% MIT License
%
% Copyright (c) 2015 TBB, 
% derived in part from work by Aimo Winkelmann (Bruker)
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.
%------------------------------------------------------------------------------

% This code is follows a similar structure to the sister (python) code
% as created by Aimo Winkelmann
%
% This matlab code also includes sections for:
% stereographic projection calculation & unit cell generation

%% Initate Script
clear
close all
home

%% Sample Setup

% triclinic example
%{
%experimental image
ExpImage.Filename='Tri_Example.png';

%lattice paramters
UCell.a =2.0;
UCell.b =3.0;
UCell.c =4.0;

UCell.alpha = 70.0; %in degrees
UCell.beta  = 100.0;
UCell.gamma = 120.0;

Detector.SampleTilt=70; %in degrees
Detector.DetectorTilt=-7;

% Euler Angles
UCell.phi1=22; %in degrees
UCell.Phi=39;
UCell.phi2=67;

% Pattern Center
Detector.DD=0.85; % in units of screen height
Detector.PCX=0.35; % in units of screen width, from left of exp. pattern
Detector.PCY=0.15;% in units of screen height, from top of exp. pattern
%}





%Ni example

ExpImage.Filename='Ni_Example3.bmp';

%lattice paramters
UCell.a=1; %(only the ratio matters)!
UCell.b=1;
UCell.c=1;

UCell.alpha=90; %in degrees
UCell.beta=90;
UCell.gamma=90;

%Euler angles
UCell.phi1=127; %in degrees
UCell.Phi=38;
UCell.phi2=273;

% Set up detector

%in degrees
Detector.SampleTilt=70;
Detector.DetectorTilt=5;

%using Bruker coordinate systems
Detector.DD=0.642; %in units of screen height
Detector.PCX=0.483; %in units of screen width, from left of exp. pattern
Detector.PCY=0.279; %in units of screen height, from top of exp. pattern

disp(UCell)
disp(Detector)
%% Load Experimental Image and Set Up Detector
%note that flipud is used as the image is loaded in ij coordinate space and
%we are going to use xy for all coordinate systems within the EBSP
ExpImage.Image=flipud(imread(ExpImage.Filename));

%in pixels
Detector.ScreenWidth=320;
Detector.ScreenHeight=230;

%use for UVW&HKL generation
HKL.max=1;

%% Establish detector
%equation 16
Detector.TotalTilt=(Detector.SampleTilt-90)-Detector.DetectorTilt;

EBSP.ScreenAspect=Detector.ScreenWidth/Detector.ScreenHeight;

EBSP.y_gn_max= Detector.PCY/Detector.DD;
EBSP.y_gn_min= -(1.0-Detector.PCY)/Detector.DD;
EBSP.x_gn_max= ((1.0-Detector.PCX)*EBSP.ScreenAspect)/Detector.DD;
EBSP.x_gn_min= -((Detector.PCX)*EBSP.ScreenAspect)/Detector.DD;

%% Establish the structure matrix

%equation 1
UCell.f=sqrt(1.0-( cosd(UCell.alpha)*cosd(UCell.alpha)...
                  +cosd(UCell.beta)*cosd(UCell.beta)...
                  +cosd(UCell.gamma)*cosd(UCell.gamma))...
    +2.0*cosd(UCell.alpha)*cosd(UCell.beta)*cosd(UCell.gamma)); 

%equation 2
UCell.ax = UCell.a * UCell.f/sind(UCell.alpha);

UCell.ay = UCell.a * (cosd(UCell.gamma)-cosd(UCell.alpha)*cosd(UCell.beta))...
           /sind(UCell.alpha);
       
UCell.az = UCell.a * cosd(UCell.beta);

%equation 3
UCell.by = UCell.b * sind(UCell.alpha);
UCell.bz = UCell.b * cosd(UCell.alpha);

%equation 4
UCell.cz = UCell.c;

%equation 5
UCell.StructureMat=[UCell.ax , 0,  0;
              UCell.ay , UCell.by, 0;
              UCell.az , UCell.bz, UCell.cz];

disp(UCell.StructureMat);
%% Establish rotation conventions

%equation 8
Rz=@(theta)[cosd(theta) sind(theta) 0;-sind(theta) cosd(theta) 0;0 0 1];

%equation 9
Rx=@(theta)[1 0 0;0 cosd(theta) sind(theta);0 -sind(theta) cosd(theta)];

%% Generate Cell Vectors

%populate the reflector list
%if this code is to be extended then the list of HKLs must be populated
%considering structure factors or a look up list
%rather than all allowable indexes - as used here

CVectors.h=-HKL.max:1:HKL.max;
[CVectors.p,CVectors.q,CVectors.r]=meshgrid(CVectors.h,CVectors.h,CVectors.h);
CVectors.pqr=[CVectors.p(:),CVectors.q(:),CVectors.r(:)];

%check for zeros & remove
CVectors.pqr=CVectors.pqr(dot(CVectors.pqr,CVectors.pqr,2)>1e-6,:);

%set as indicies for pqr for HKL and UVW (in crystal coords)
%this is a slightly different order to the sister python code
CVectors.HKL=CVectors.pqr;
CVectors.UVW=CVectors.pqr;

disp(CVectors.HKL)

%% Generate formal rotations & coordinate transforms as used

%U.S = rotation for detector & sample conversion
U.S=Rx(Detector.TotalTilt);

%equation 10
%U.O = orientation of sample
U.O=Rz(UCell.phi2)*Rx(UCell.Phi)*Rz(UCell.phi1);

%U.At = transpose of structure matrix as used for UVW conversions
U.At=transpose(UCell.StructureMat);

%U.AStar = recriprical structure matrix as used for HKL conversions
U.Astar=inv(UCell.StructureMat);

%U.K and U.Kstar = transforming the orientation into the detector plane
%K = for UVW; Kstar = for HKL
U.K         =   U.At*U.O*U.S;
U.Kstar     =   U.Astar*U.O*U.S;

disp(U.At)
disp(U.Astar)
disp(U.K)
disp(U.Kstar)

%% Convert UVW and HKL into the detector frame
%equation 13 (with Os included within U.K - making this more like equation 17)
UVW.D=CVectors.UVW*U.K;

%equation 14 (with Os included within U.K)
HKL.D=CVectors.HKL*U.Kstar;

%% Construct the EBSP

%UVW - [X,Y,Z]
UVW.X=transpose(UVW.D(:,1));
UVW.Y=transpose(UVW.D(:,2));
UVW.Z=transpose(UVW.D(:,3));

%UVW Gn ratios - used for labelling the coords on the EBSP
UVW.x_gn=UVW.X./UVW.Z;
UVW.y_gn=UVW.Y./UVW.Z;

%HKL - [X,Y,Z]
HKL.X=transpose(HKL.D(:,1));
HKL.Y=transpose(HKL.D(:,2));
HKL.Z=transpose(HKL.D(:,3));

%HKL - needed for hessian calculations
HKL.r=sqrt(HKL.X.^2+HKL.Y.^2+HKL.Z.^2);
HKL.kai=atan2(HKL.Y,HKL.X);
HKL.theta=acos(HKL.Z./HKL.r);

%Hessian construction
Hess.R_Hesse=10; %radius of the Hessian
Hess.d_Hesse=tan(0.5*pi-HKL.theta);
Hess.alpha_Hesse=acos(Hess.d_Hesse./Hess.R_Hesse);

Hess.alpha1_hkl=HKL.kai-pi+Hess.alpha_Hesse;
Hess.alpha2_hkl=HKL.kai-pi-Hess.alpha_Hesse;

%[C1x,C1y] to [C2x,C2y] are the coords on the screen
Hess.C1x=Hess.R_Hesse.*cos(Hess.alpha1_hkl);
Hess.C1y=Hess.R_Hesse.*sin(Hess.alpha1_hkl);
Hess.C2x=Hess.R_Hesse.*cos(Hess.alpha2_hkl);
Hess.C2y=Hess.R_Hesse.*sin(Hess.alpha2_hkl);

%% Plot the EBSP

%Plot the example image first

%establish the coordinate systems
%from pixels --> Gnomonic
EBSP.x_img=1:size(ExpImage.Image,2);
EBSP.y_img=1:size(ExpImage.Image,1);
EBSP.x_img=(EBSP.x_gn_max-EBSP.x_gn_min)*(EBSP.x_img-1)/max(EBSP.x_img)+EBSP.x_gn_min;
EBSP.y_img=(EBSP.y_gn_max-EBSP.y_gn_min)*(EBSP.y_img-1)/max(EBSP.y_img)+EBSP.y_gn_min;

%plot the figure

%example EBSP
figure;
imagesc(EBSP.x_img,EBSP.y_img,ExpImage.Image);
xlim([EBSP.x_gn_min,EBSP.x_gn_max]);
ylim([EBSP.y_gn_min,EBSP.y_gn_max]);
axis xy;
colormap('gray')

%plot the bands
num_HKL=size(Hess.C1x,2);
for n=1:num_HKL
    if HKL.Z(n)>0 %if upper hemisphere
        hold on
        plot([Hess.C1x(n) Hess.C2x(n)],[Hess.C1y(n) Hess.C2y(n)],'-k','LineWidth',2);
    end
end

%label the zone axes
for n=1:num_HKL
   hold on
   if UVW.Z(n)>0
       temp.Text1=sprintf('[%02.0f, %02.0f, %02.0f]',CVectors.HKL(n,:));
       temp.ah=annotation('textbox','String',temp.Text1,'Color','w','EdgeColor','none');
       
       set(temp.ah,'parent',gca); %changes the annotation to belong the the parent axes - required for coords to work
       set(temp.ah,'position',[UVW.x_gn(n) UVW.y_gn(n) 0.3 .1]); %changes the position to the correct position
       scatter(UVW.x_gn(n), UVW.y_gn(n),30,'w') %puts a white circle on the axes locations
   end
end

%Plot the pattern centre
% scatter(0,0,'rx');
scatter(0,0,50,'r*');
temp.pc=annotation('textbox','String','PC','Color','w','EdgeColor','none');
       
set(temp.pc,'parent',gca); %changes the annotation to belong the the parent axes - required for coords to work
set(temp.pc,'position',[0 0 0.3 .1]); %changes the position to the correct position

%label the axes
xlabel('X / Z');
ylabel('Y / Z');


axis image
xlim([-1.5 1.5]);
ylim([-1.5 1.5]);
%clean up tempvars
clear temp

%% Calculate in the sample coordinate system
%U2.K and U2.Kstar = transforming the orientation into the sample plane
%K = for UVW; Kstar = for HKL
U2.K         =   U.At*U.O;
U2.Kstar     =   U.Astar*U.O;

%% Stereographic Calculation

% create the plotting family
HKL_family=[1 1 1;-1 1 1;1 -1 1;1 1 -1;-1 -1 1;-1 1 -1;1 -1 -1;-1 -1 -1];

% transform to sample coordinate system
HKL_family_s=HKL_family*U2.Kstar;

% convert to unit length
HKL_family_s_unit=HKL_family_s./repmat(sqrt(dot(HKL_family_s,HKL_family_s,2)),1,3);

% convert to stereo
stereo.pole_sign=sign(HKL_family_s_unit(:,3));
% in the equatorial plane = up
stereo.pole_sign(stereo.pole_sign==0)=1;
% repeat to solve
stereo.pole_sign=repmat(stereo.pole_sign,1,2);

%solve for the projections
stereo.UVW=HKL_family_s_unit(:,1:2).*stereo.pole_sign./(repmat(HKL_family_s_unit(:,3),1,2)+stereo.pole_sign);

%% Plot the stereographic projection
figure;

%generate the circle for the stereogram
stereo.theta=0:360;
stereo.c_x=cosd(stereo.theta);
stereo.c_y=sind(stereo.theta);
plot(stereo.c_x,stereo.c_y,'k');
hold on;
plot(stereo.c_x([1,181]),stereo.c_y([1,181]),'k');
plot(stereo.c_x([91,271]),stereo.c_y([91,271]),'k');

%plot the [001] stereogram
scatter(stereo.UVW(HKL_family_s(:,3)>0,1),stereo.UVW(HKL_family_s(:,3)>0,2),'r','filled');
scatter(stereo.UVW(HKL_family_s(:,3)<0,1),stereo.UVW(HKL_family_s(:,3)<0,2),'r');

axis equal;

%% Plot a unit cube in the sample frame

%define the three basis vectors for the cube
Cube.poi_cen=[0 0 0];
Cube.a1=[1 0 0]*U2.K;
Cube.a2=[0 1 0]*U2.K;
Cube.a3=[0 0 1]*U2.K;

% % uncomment if you want to plot in the detector frame
% a1=[1 0 0]*U.K;
% a2=[0 1 0]*U.K;
% a3=[0 0 1]*U.K;

%create the faces from these vectors
Cube.face_1a=[Cube.poi_cen;
    Cube.a1+Cube.poi_cen;
    Cube.a1+Cube.a2+Cube.poi_cen;
    Cube.a2+Cube.poi_cen;
    Cube.poi_cen];
Cube.face_1b=Cube.face_1a+repmat(Cube.a3,size(Cube.face_1a,1),1);

Cube.face_2a=[Cube.poi_cen;
    Cube.a1+Cube.poi_cen;
    Cube.a1+Cube.a3+Cube.poi_cen;
    Cube.a3+Cube.poi_cen;
    Cube.poi_cen];
Cube.face_2b=Cube.face_2a+repmat(Cube.a2,size(Cube.face_2a,1),1);

Cube.face_3a=[Cube.poi_cen;
    Cube.a2+Cube.poi_cen;
    Cube.a2+Cube.a3+Cube.poi_cen;
    Cube.a3+Cube.poi_cen;
    Cube.poi_cen];
Cube.face_3b=Cube.face_3a+repmat(Cube.a1,size(Cube.face_3a,1),1);

figure;
patch(Cube.face_1a(:,1),Cube.face_1a(:,2),Cube.face_1a(:,3),[1 0 0],'EdgeColor','k','FaceColor','r');
axis equal;
hold on
patch(Cube.face_1b(:,1),Cube.face_1b(:,2),Cube.face_1b(:,3),[1 0 0],'EdgeColor','k','FaceColor','r');
patch(Cube.face_2a(:,1),Cube.face_2a(:,2),Cube.face_2a(:,3),[1 0 0],'EdgeColor','k','FaceColor','g');
patch(Cube.face_2b(:,1),Cube.face_2b(:,2),Cube.face_2b(:,3),[1 0 0],'EdgeColor','k','FaceColor','g');
patch(Cube.face_3a(:,1),Cube.face_3a(:,2),Cube.face_3a(:,3),[1 0 0],'EdgeColor','k','FaceColor','b');
patch(Cube.face_3b(:,1),Cube.face_3b(:,2),Cube.face_3b(:,3),[1 0 0],'EdgeColor','k','FaceColor','b');

