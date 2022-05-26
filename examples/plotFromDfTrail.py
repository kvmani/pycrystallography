import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
import webbrowser

df = pd.read_csv("df_difraftionPatternFigure.csv")
dfTmp = df.copy()
dfTmp=dfTmp[dfTmp["intensity"]>1e-3]
#df.round({'dogs': 1, 'cats': 0})
dfTmp = dfTmp.round({"XY":3, "dSpacing":3})
#dfTmp["XY"] = dfTmp["XY"]
table = dfTmp.pivot(index="SpotGroupId", columns="PhaseName",
                 values=["SpotHkl","XY","dSpacing","intensity"])
print(table)
htmlFile = os.path.join(r"../tmp", "trial.html")
fileName = f'file:///'+os.path.abspath(htmlFile)
table.to_html(htmlFile)
webbrowser.open_new_tab(fileName)

print(df)
exit(-10)
fc = colors.to_rgba('lightgrey')
ec = colors.to_rgba('black')
fc = fc[:-1]+(0.5,)

fig = plt.figure()
ax = fig.add_subplot(111, )

# for angle in range(0,360,30):
#     label = str(angle)
#     LabelPosition = (np.cos(angle*np.pi/180)*100,np.sin(angle*np.pi/180)*100 )
#     ax.annotate(label, xy=(0, 0), xytext=LabelPosition,
#                                          textcoords='offset points', ha='center', va='center',
#                                         rotation = -angle,rotation_mode='anchor',
#                                          bbox=dict(facecolor=fc, edgecolor=ec, pad=0.0),
#                                         arrowprops=dict(arrowstyle="->")
#                                         )


for index, row in df.iterrows():
    if not row["absentSpot"]:
        x, y = np.fromstring(row["XY"].replace('[','').replace(']',''), dtype = float, sep=',')
        if row["showMarker"]:
            marker, markerColor, markerSize = row["markerStyle"], row["markerColor"], row["markerSize"]
            ax.scatter(x,y,marker=marker,s=markerSize, facecolors='none', edgecolors=markerColor)
    if row["markSpot"]:
        if not row["absentSpot"]:
            label,LabelPosition,angle = row["Label"], \
                                  30*(np.fromstring(row["LabelPosition"].replace('[', '').replace(']', ''), dtype=float, sep=',')),\
                                    row["LableAngle"]

            if row["NumberOfOverLappingSpots"]<2:
                LabelPosition, rotation = [0,15],0

            ax.annotate(label, xy=(x,y), xytext=LabelPosition,
                         textcoords='offset points', ha='center', va='center',
                        rotation = -0,
                         bbox=dict(facecolor=fc, edgecolor=ec, pad=0.0),
                        arrowprops=dict(arrowstyle="->")
                        )

# markerStyle, markerSize = [item for item in df["markerStyle"].tolist()], [item for item in df["markerSize"].tolist()]
ax.plot([0.25,0.25,-0.25,-0.25],[-0.25,0.25,0.25,-0.25],'-r')

ax.autoscale()
ax.legend()
ax.set_aspect('equal')
ax.axis('equal')
#plt.scatter(x,y,s=markerSize,)
plt.show()
print(df)
htmlFile = os.path.join(r"../tmp", "trial.html")
fileName = f'file:///'+os.path.abspath(htmlFile)
df.to_html(htmlFile)
webbrowser.open_new_tab(fileName)