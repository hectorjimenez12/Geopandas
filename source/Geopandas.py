# -*- coding: utf-8 -*-
"""
Created on Abril 2022
Mapas creados con geopandas
@author: Hector Jimenez
"""

import pandas as pd
import geopandas as gpd 
from shapely.geometry import Point
from shapely.geometry import Polygon
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib 
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import random
import matplotlib.cbook as cbook
#from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patheffects as pe


matplotlib.rcParams.update({'font.size': 13})
mainfolder = 'C:/Users/hecto/OneDrive/Escritorio/AMTC/2023/MOP_MANUAL_QUEBRADAS/Etapa_Analisis_Hidrologico/Datos/Camels/'
foldershp = mainfolder + 'shapefiles/'  
os.chdir(mainfolder + 'Python_Fun/Classes/')
from geodata import geodata, raster, dem, shape
from geodict import geodict
os.chdir(mainfolder)

worldgeo = shape(name=['Mundo'],variable='paises')
chilereg = shape(name = ['Regiones'],variable = 'Cuencas', source = str(foldershp + 'Regiones/regionesGEO.shp'))
chile_simple = worldgeo.SubShape(col='name',idsel='Chile')
chile_simple.plot()
chilereg.shp.plot()


regiones = chilereg.shp
regiones.codregion
regiones['CEreg'] = np.array([np.nan]*5 + [0.009, np.mean([0.025, 0.078, 0.080]),0.008 ] + [np.nan]*3 + [0.28,np.nan,np.nan,0.39,0.31] + [np.nan])
regiones.columns


dfestaciones = pd.read_csv('DF_RFC.csv')
regiones.plot(column='CEreg',cmap='jet')







''' UTILIZACION DE CLASE GEO '''
os.getcwd()
os.chdir(mainfolder)
df = pd.read_csv('DF_RFC.csv')

import math 
from matplotlib import gridspec
jet=plt.get_cmap('coolwarm')
matplotlib.rcParams.update({'font.size': 5})
shp = gpd.read_file('all_basins/camels_cl_boundaries/camels_cl_boundaries.shp')

cols_clus = ['blue','red','orange','green','purple','brown','pink','yellow']

ylims = [ [ (df.lat.min() - df.lat.max())/2 +df.lat.max() , df.lat.max()   ],
          [df.lat.min()   , (df.lat.min() - df.lat.max())/2 +df.lat.max() ]               ]
xlims = [[-73,-67],[-76,-68]]

Ncols = 2
#fig, ax = plt.subplots(figsize = (2,2))
fig = plt.figure(figsize=(6, 6)) 
cm = plt.cm.get_cmap('RdYlBu')
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1] ) 
k=0
#shp.boundary.plot(ax=ax,linewidth=0.02,color='black')
for iplot in range(Ncols):
    
    ax =  plt.subplot(gs[iplot])
    print(iplot)
    
    chilereg.shp.boundary.plot(ax=ax,linewidth=0.1,color='black')
    regiones.plot(ax=ax,column='CEreg',cmap = cm,vmin=0,vmax=1)
    #unique_clus = np.unique(np.array(df.cluster))
    sc = ax.scatter(df.lon,
               df.lat,
               alpha      = 1,#zorder=1,
               linewidths = 0.1,
               edgecolor  ="black",
               c= df.rfc , s=15,
               cmap = cm,vmin=0,vmax=1)
    k +=1
    ax.set_xlim(xlims[iplot])
    ax.set_ylim(ylims[iplot])
    #ax.legend(loc = 0, prop = {'size':3} )
fig.tight_layout()
plt.colorbar(sc)
plt.subplots_adjust(wspace=-0.6, hspace=0)
fig.savefig('Figuras/CE.pdf',dpi=1500,bbox_inches='tight')




''' 3D PLOT FOR CLUSTER AGRUPATIONS'''
#import statistics
#statistics.correlation(df.peak,df.rfc)

cor = df.corr()

from mpl_toolkits import mplot3d
uparrow =   '\u2191'
downarrow = '\u2193'

Desc_Cluster = [uparrow + 'Lag, ' + uparrow +'Duracion, '+downarrow+'Peak', #$\mathregular{\downarrow} Peak  $
                downarrow + 'Lag, ' + uparrow +'Duracion, '+'-Peak',
                downarrow + 'Lag, ' + uparrow +'Duracion, '+downarrow + 'Peak',
                downarrow + 'Lag, ' + downarrow +'Duracion, '+downarrow+'Peak',
                downarrow + 'Lag, ' + uparrow +'Duracion, '+uparrow+uparrow+'Peak',
                downarrow + 'Lag, ' + uparrow +'Duracion, '+uparrow+uparrow+'Peak']

Desc_Cluster= [' ']*8

fig = plt.figure()
k=0
ax = fig.add_subplot(111,projection = '3d')
#plotting all the points
unique_clus = np.unique(np.array(df.cluster))
for i in unique_clus:
    if not math.isnan(i) :
        id_clus = np.where(df.cluster == i)
        ax.scatter(df.rfc[id_clus[0]],
                   df.lag[id_clus[0]],
                   df.lat[id_clus[0]],
                   alpha= 1,#zorder=1,
                   linewidths = 0.1,
                   edgecolor="black",
                   c= cols_clus[k] , label= str(int(i)) + '(' + Desc_Cluster[k] +')',
                   s=15)
        k+=1
#ax.set_xlim([0,200])
#ax.set_zlim([0,100])
#ax.scatter(xcod,ycod,zcod,'x-')
plt.legend(numpoints=1, bbox_to_anchor=(0.5, 0.7), title="Id Cluster")
ax.set_xlabel("RFC [-]")
ax.set_ylabel("Lag [-]")  #"Max Peak [mm/dia]")
ax.set_zlabel("Lat [-]")
#ax.view_init(0, 90)
#plt.savefig('Figuras/Cluster3D__90grad.pdf', bbox_inches='tight')
plt.savefig('Figuras/Cluster3D_6.pdf', bbox_inches='tight')




matplotlib.rcParams.update({'font.size': 8})
fig = plt.figure()
k=0
ax = fig.add_subplot(111)
#plotting all the points
unique_clus = np.unique(np.array(df.cluster))
for i in unique_clus:
    if not math.isnan(i) :
        id_clus = np.where(df.cluster == i)
        ax.scatter(df.rfc[id_clus[0]],
                   df.lag[id_clus[0]],
                   alpha= 1,#zorder=1,
                   linewidths = 0.1,
                   edgecolor="black",
                   c= cols_clus[k] , label= str(int(i)) + '(' + Desc_Cluster[k] +')',
                   s=15)
        k+=1
#ax.set_xlim([0,1])
#ax.set_zlim([0,100])
#ax.scatter(xcod,ycod,zcod,'x-')
plt.legend(numpoints=1, bbox_to_anchor=(0.5, 0.7), title="Id Cluster")
ax.set_xlabel("Coef. Escorrentia tormenta [-]")
ax.set_ylabel("Lag [-]")  #"Max Peak [mm/dia]")
plt.savefig('Figuras/Clusterxy.pdf', bbox_inches='tight')



matplotlib.rcParams.update({'font.size': 8})
fig = plt.figure()
k=0
ax = fig.add_subplot(111)
#plotting all the points
unique_clus = np.unique(np.array(df.cluster))
for i in unique_clus:
    if not math.isnan(i) :
        id_clus = np.where(df.cluster == i)
        ax.scatter(df.lag[id_clus[0]],
                   df.rfc[id_clus[0]],
                   alpha= 1,#zorder=1,
                   linewidths = 0.1,
                   edgecolor="black",
                   c= cols_clus[k] , label= str(int(i)) + '(' + Desc_Cluster[k] +')',
                   s=15)
        k+=1
#ax.set_xlim([0,1])
#ax.set_zlim([0,100])
#ax.scatter(xcod,ycod,zcod,'x-')
plt.legend(numpoints=1, bbox_to_anchor=(0.5, 0.7), title="Id Cluster")
ax.set_ylabel("Coef. Escorrentia tormenta [-]")
ax.set_xlabel("Lag (Distancia Max PP - Max Esc. Dir) / Tc [-]")  #"Max Peak [mm/dia]")
plt.savefig('Figuras/Clusterxy.pdf', bbox_inches='tight')



matplotlib.rcParams.update({'font.size': 8})
fig = plt.figure()
k=0
ax = fig.add_subplot(111)
#plotting all the points
unique_clus = np.unique(np.array(df.cluster))
for i in unique_clus:
    if not math.isnan(i) :
        id_clus = np.where(df.cluster == i)
        ax.scatter(df.rfc[id_clus[0]],
                   df.peak[id_clus[0]],
                   alpha= 1,#zorder=1,
                   linewidths = 0.1,
                   edgecolor="black",
                   c= cols_clus[k] , label= str(int(i)) + '(' + Desc_Cluster[k] +')',
                   s=15)
        k+=1
#ax.set_xlim([0,1])
#ax.set_zlim([0,100])
#ax.scatter(xcod,ycod,zcod,'x-')
plt.legend(numpoints=1, bbox_to_anchor=(0.5, 0.7), title="Id Cluster")
ax.set_xlabel("Coef. Escorrentia tormenta [-]")
ax.set_ylabel("Peak ")  #"Max Peak [mm/dia]")
plt.savefig('Figuras/Clusteryz.pdf', bbox_inches='tight')




##################################################################################



fig , ax = plt.subplots(ncols=3,figsize=(2,3),sharey=True)
shp.plot(ax=ax[0],linewidth=0.05,column='Lag',legend=True,
         cmap='PiYG',legend_kwds={'label': "Lag [días]",
                          'orientation': "vertical"})
shp.boundary.plot(ax=ax[0],linewidth=0.02,color='black')
shp.plot(ax=ax[1],linewidth=0.05,column='Coef_Esc_Max',legend=True,
         cmap='PiYG',legend_kwds={'label': "Coef.\nEscorrentía\nMax. [-]",
                          'orientation': "vertical"})
shp.boundary.plot(ax=ax[1],linewidth=0.02,color='black')
shp.plot(ax=ax[2],linewidth=0.05,column='Max_RF',legend=True,
         cmap='PiYG',legend_kwds={'label': "Max.\nEscorrentía",
                          'orientation': "vertical"})
shp.boundary.plot(ax=ax[2],linewidth=0.02,color='black')
fig.tight_layout()
fig.savefig('Lag.pdf',dpi=500)

import matplotlib.colors as colors

fig = plt.figure()
gs = fig.add_gridspec(1, 3, hspace=0, wspace=0)
(ax1, ax2, ax3) = gs.subplots(sharey='row')
fig.suptitle('Indices hidrologicos en cuencas Camels')
shp.plot(ax=ax1,linewidth=0.05,column='Lag',legend=True,
         norm = colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=shp.Lag.min(), vmax=shp.Lag.max(), base=10),
         cmap='PiYG',legend_kwds={'label': "Lag [días]",      'orientation': "vertical"})
shp.plot(ax=ax2,linewidth=0.05,column='Coef_Esc_Max',legend=True,
         cmap='PiYG',norm=matplotlib.colors.LogNorm(vmin=shp.Coef_Esc_Max.min(), vmax=shp.Coef_Esc_Max.max()),
         legend_kwds={'label': "Coef.\nEscorrentía\nMax. [-]",
                          'orientation': "vertical"})
shp.plot(ax=ax3,linewidth=0.05,column='Max_RF',legend=True,
         norm=matplotlib.colors.LogNorm(vmin=shp.Max_RF.min(), vmax=shp.Max_RF.max()),
         cmap='PiYG',legend_kwds={'label': "Max.\nEscorrentía",
                          'orientation': "vertical"})
fig.savefig('Lag_log.pdf',dpi=500)
for ax in fig.get_axes():
    ax.label_outer()









# GEO OBJECT
ylims = [-24,-21.5]
xlims = [-71,-68]
geoNorte =        Geo(name='Subsubcuencas Rio Salado y Rio San Pedro')
geoNorte.addData(namedata='Red hidrografica',rutashp='red_hidrografica/red_hidrografica_geo.shp')
geoNorte.addData(namedata='Regiones',rutashp='Regiones/regionesGEO.shp')
geoNorte.addData(namedata='Operacion Minera',rutashp='FaenasMineras/Puntos_MinerasGeo.shp')
geoNorte.addData(namedata='Subsubcuencas',rutashp='SubsubcuencasBNA/subsubcuencasGeo.shp')
geoNorte.addData(namedata='Cuencas',rutashp='cl_cuencas_hidrograficas_geo/cl_cuencas_hidrograficas_geoPolygon.shp')
geoNorte.addData(namedata='Pozos',rutashp='EstacionesNivelesPozos/PozosGeo.shp')
geoNorte.addData(namedata='Estaciones Fluviometricas',rutashp='EstacionesFluviometricas/fluviometricasGEO.shp')
#'_nolegend_' para un nombre que no tenga legend

#idSubcuencas1 = np.where( [ ('Loa' in geoNorte.dic['Subsubcuencas']['NOM_SSUBC'][i] or 'Salado' in geoNorte.dic['Subsubcuencas']['NOM_SSUBC'][i]) for i in range(len(geoNorte.dic['Subsubcuencas'])) ] )[0]
#idSubcuencas1 = np.where( [ ('Salado' in geoNorte.dic['Subsubcuencas']['NOM_SSUBC'][i]) for i in range(len(geoNorte.dic['Subsubcuencas'])) ] )[0]
#idSub1 =  list(geoNorte.dic['Subsubcuencas']['NOM_SSUBC'][idSubcuencas1]) 
#shapeRegiones = geoNorte.dic['Regiones']
#idSubcuencas1 = np.where( [ ('Antofagasta' in shapeRegiones['NOM_REG'][i]) for i in range(len(shapeRegiones)) ] )[0]
#idSub1 =  list(shapeRegiones['NOM_REG'][idSubcuencas1])

#matplotlib.pyplot.xticks(fontsize=20)
#matplotlib.pyplot.yticks(fontsize=20)

CuencaLoa = geoNorte.GetSubShape(nameShape = 'Estaciones Fluviometricas',
                                    columnSearch = 'NOMBRE',
                                    idSelect = ['RIO SAN PEDRO EN PARSHALL 2 (CODELCO)',
                                                'RIO SAN PEDRO EN PARSHALL N"2 (BT. CHILEX)'])



geoNorte.AddLabelShape(listlabel=[['Operacion Minera','Mina',['Chuquicamata'] ,[-0.15,0.15  ] , None ],
                                   ['Subsubcuencas','NOM_SSUBC',['Rio Salado'] ,[0,0.15  ], 'Cuenca_Propuesta' ],
                                   ['Subsubcuencas','NOM_SSUBC',['Rio San Pedro'] ,[0.2,0.13  ], None ],
                                   ['Estaciones Fluviometricas','NOMBRE',['RIO SAN PEDRO EN PARSHALL N"2 (BT. CHILEX)'] ,[0,0.35  ], None ],
                                   ['Estaciones Fluviometricas','NOMBRE',['RIO SAN PEDRO EN PARSHALL N"1'] ,[0,0.25  ], None ],
                                   
                                   ['Estaciones Fluviometricas','NOMBRE',['RIO TOCONCE ANTES REPRESA SENDOS'] ,[0,-0.45  ], None ],
                                   ['Estaciones Fluviometricas','NOMBRE',['RIO SALADO EN SIFON AYQUINA'] ,[0,-0.3  ], None ],
                                    ['Estaciones Fluviometricas','NOMBRE',['RIO SALADO A. J. LOA'] ,[0,-0.1  ], None ]] )


#geoNorte.dic['Estaciones Fluviometricas']['NOMBRE']



#['Subsubcuencas','NOM_SSUBC',['Rio Salado','Rio San Pedro'] ,[0,0.11  ], 'Cuenca_Propuesta' ]
geoNorte.PlotShapefiles(titulo = 'Mapa Cuenca Rio Salado en Operacion Cluster Calama ' , 
                        colors=['blue','black','brown','red','black','yellow','orange','brown','green','red'] +['darkorange']*5 ,
                        shapeNameLim= 'Cuencas',linetype= ['dashed']+['solid']*(7) + ['dashed']+['solid']*6 ,
                        linesW=[0.3,1,25,0.5,0.5,25,25,10,10,10] +[8]*5,
                        sizeLeg=7,sizeFig=[6,5],#scale=False,
                        idShapesLim= ['Rio Loa'] ,atributeSearchLim='NOMBRE',
                        additionalText=[[ 0.92,0.85, 'Argentina' ,'vertical','large','black' ]], faceColor='lightgray')





geoNorte.AddLabelShape(listlabel=[['Operacion Minera','Mina',['Andina'] ,[0.1,0.1] , None ] ,
                                  ['Subsubcuencas','NOM_SSUBC',['Rio Blanco'] ,[-0.5,-0.04 ], 'Cuenca_Propuesta'  ],
                                  ['Estaciones Fluviometricas','NOMBRE',['RIO BLANCO EN RIO BLANCO'] ,[-0.5,0.05  ], None ],
                                  ['Estaciones Fluviometricas','NOMBRE',['RIO BLANCO ANTES JUNTA RIO DE LOS LEONES'] ,[-0.7,0.03  ], None ]] )

geoNorte.PlotShapefiles(titulo = 'Mapa Cuenca Rio Blanco en Operacion Minera Andina' , 
                        colors=['blue','black','brown','red','black','yellow','orange','brown','green','darkorange','darkorange'],
                        shapeNameLim= 'Cuencas',linetype= ['dashed']+['solid']*7 + ['dashed','solid','solid']  ,
                        linesW=[0.3,1,25,0.5,0.5,25,25,10,10,8,8] ,
                        sizeLeg=8,sizeFig=[6,4.5],#scale=False,
                        idShapesLim= ['Rio Aconcagua'] ,atributeSearchLim='NOMBRE',
                        additionalText=[[ 0.95,0.7, 'Argentina' ,'vertical','large','black' ],
                                        [ 0.03,0.65, 'O. Pacifico' ,'vertical','medium','blue' ]], faceColor='lightgray') 






#%%

folder = 'G:/.shortcut-targets-by-id/1IopAan1jtVEcZh6gcCmTiHwWztAL6dij/07 Guía de disponibilidad hidrica futura ante escenarios de CC/05 test/Entregable5/'
os.chdir(folder)
mainfolder = 'G:/.shortcut-targets-by-id/1IopAan1jtVEcZh6gcCmTiHwWztAL6dij/07 Guía de disponibilidad hidrica futura ante escenarios de CC/'
foldershp = mainfolder + '01 Base de datos/Info_Geo/'  
folderDem = 'D:/AMTC/CODELCO/Entregable5/BaseDatos/'
folderClass = 'G:/Mi unidad/Py_Classes/2022/Classes/'
os.chdir(folderClass)

from geodata import geodata, raster, dem, shape
from geodict import geodict

saladodem = raster(name ='DEM Rio Salado', variable = 'Altura [m.s.n.m]',
                source = str(folderDem + 'demRsaladoAP.tif') ,typerst = 'Gdal' , cmap = 'RdBu')
saladodem = dem(name ='DEM Rio Salado', variable = 'Altura [m.s.n.m]',
                source = str(folderDem + 'demRsaladoAP.tif') , cmap = 'RdBu')
blancodem = dem(name ='DEM Rio Blanco', variable = 'Altura [m.s.n.m]',
                source = str(folderDem + 'demRblancoAP.tif'), cmap = 'RdBu' )
saladoshp = shape(name = ['Rio Salado'],variable = 'Rio Salado',colName = ['NOM_SSUBC'],
                  source = str(foldershp + 'SubsubcuencasBNA/subsubcuencasGeo.shp') )
blancoshp = shape(name = ['Rio Blanco', '05402'],variable = 'Rio Blanco',colName = ['NOM_SSUBC', 'COD_SSUBC' ],
                  source = str(foldershp + 'SubsubcuencasBNA/subsubcuencasGeo.shp') )
saladodem.clip_extend(saladoshp)
blancodem.clip_extend(blancoshp)
blancodem.plotr(shp = blancoshp.shp)

''' PLOT DE COLORBAR'''
os.getcwd()
os.chdir('G:/.shortcut-targets-by-id/1IopAan1jtVEcZh6gcCmTiHwWztAL6dij/07 Guía de disponibilidad hidrica futura ante escenarios de CC/01 Base de datos/RedHidrografica_Cuencas/')

salado = Geo(name = 'Rio Salado', cmap='Blues')
salado.dic
#salado.addData(namedata='Cauces',rutashp='red_hidrografica/red_hidrografica_geo.shp')
salado.addData(namedata='Pozos DGA',rutashp='EstacionesNivelesPozos/PozosGeo.shp')
salado.addData(namedata='Est. Metereologicas',rutashp='EstacionesMeteorologicas/metereologicasGEO.shp')
salado.addData(namedata='Est. Fluviometricas',rutashp='EstacionesFluviometricas/fluviometricasGEO.shp')
salado.addData(namedata='Rio Salado', shp = saladoshp.shp)
#salado.addData(namedata='Subsubcuencas',rutashp='SubsubcuencasBNA/subsubcuencasGeo.shp')
#salado.dic['Rio Salado'].geom_type


salado.PlotShapefiles(titulo = 'Rio Salado ' , 
                        colors=['orange','blue','red','black','black','yellow','orange','brown','green','red'] +['darkorange']*5 ,
                        shapeNameLim= 'Rio Salado',linetype= ['dashed']+['solid']*(7) + ['dashed']+['solid']*6 ,
                        linesW=[100,100,100,2] ,#limX=[-80,-70], limY=[-40,-35],
                        sizeLeg=10,sizeFig=[6,5],#scale=False,
                        idShapesLim= ['Rio Salado'] ,atributeSearchLim='NOM_SSUBC',rst=saladodem)

salado.removekey('Rio Salado')
salado.removekey('Blanco')
salado.addData(namedata='Rio Blanco', shp = blancoshp.shp)

salado.PlotShapefiles(titulo = 'Rio Blanco ' , 
                        colors=['orange','blue','red','black','black','yellow','orange','brown','green','red'] +['darkorange']*5 ,
                        shapeNameLim= 'Rio Blanco',linetype= ['dashed']+['solid']*(7) + ['dashed']+['solid']*6 ,
                        linesW=[100,100,100,2] ,#limX=[-80,-70], limY=[-40,-35],
                        sizeLeg=10,sizeFig=[6,5],#scale=False,
                        idShapesLim= ['Rio Blanco'] ,atributeSearchLim='NOM_SSUBC',rst=blancodem)



#geoNorte =        Geo(name='Subsubcuencas Rio Salado y Rio San Pedro')
#geoNorte.addData(namedata='Red hidrografica',rutashp='red_hidrografica/red_hidrografica_geo.shp')
#geoNorte.addData(namedata='Regiones',rutashp='Regiones/regionesGEO.shp')
#geoNorte.addData(namedata='Operacion Minera',rutashp='FaenasMineras/Puntos_MinerasGeo.shp')
#geoNorte.addData(namedata='Subsubcuencas',rutashp='SubsubcuencasBNA/subsubcuencasGeo.shp')
#geoNorte.addData(namedata='Cuencas',rutashp='cl_cuencas_hidrograficas_geo/cl_cuencas_hidrograficas_geoPolygon.shp')
#geoNorte.addData(namedata='Pozos',rutashp='EstacionesNivelesPozos/PozosGeo.shp')
#geoNorte.addData(namedata='Estaciones Fluviometricas',rutashp='EstacionesFluviometricas/fluviometricasGEO.shp')


#geoNorte.PlotShapefiles(titulo = 'Mapa Cuenca Rio Salado en Operacion Cluster Calama ' , 
#                        colors=['blue','black','brown','red','black','yellow','orange','brown','green','red'] +['darkorange']*5 ,
#                        shapeNameLim= 'Subsubcuencas',linetype= ['dashed']+['solid']*(7) + ['dashed']+['solid']*6 ,
#                        linesW=[0.3,1,25,0.5,0.5,25,25,10,10,10] +[8]*5,
#                        sizeLeg=7,sizeFig=[6,5],#scale=False,
#                        idShapesLim= ['Rio Salado'] ,atributeSearchLim='NOM_SSUBC')
                        #additionalText=[[ 0.92,0.85, 'Argentina' ,'vertical','large','black' ]], faceColor='lightgray')











#%%

#s = geoNorte.GetSubShape(nameShape = 'Subsubcuencas', columnSearch = 'NOM_SSUBC', idSelect = ['Rio Salado','Rio San Pedro'])
shapeRegiones = geoNorte.dic['Subsubcuencas']
type(shapeRegiones['coords'][0])
shapeRegiones['coords'][0][0][1]

tuple([1,1])
#Labels

cx = [ shapeRegiones['geometry'][ipol].representative_point().coords[:][0][0]  for ipol in range(len(shapeRegiones))  ]
cx = [ shapeRegiones['geometry'][ipol].representative_point().coords[:][0][1]  for ipol in range(len(shapeRegiones))  ]


shapeRegiones['coords'] = shapeRegiones['geometry'].apply(lambda x: x.representative_point().coords[:])
shapeRegiones['coords'] = [coords[0] for coords in shapeRegiones['coords']]
shapeRegiones['coordsText'] = [ (coords[0] +0.5 ,coords[1]+0.5 ) for coords in shapeRegiones['coords']]

arrowprops=dict(arrowstyle='->', color='blue', linewidth=1)
shapeRegiones.plot()
for i in range(len(shapeRegiones)):
    plt.annotate(text=shapeRegiones['NOM_SSUBC'][i], xy=shapeRegiones['coords'][i],
                 xytext = shapeRegiones['coordsText'][i],
                 horizontalalignment='center',arrowprops=arrowprops,fontsize=1)







# CMAP 
top = plt.get_cmap('Blues', 8)
bottom = plt.get_cmap('BrBG', 50)
bottom = ListedColormap(bottom(np.linspace(0.2, 0.3, 2)))
top = ListedColormap(top(np.linspace(0.7, 1, 8)))
newcolors =  np.vstack((bottom(np.linspace(0, 1, 1)),top(np.linspace(0, 1, 8))))
newcmp = ListedColormap(newcolors, name='greens')





ppChile = Geo(name='Precipitacion anual de Chile 1979-2014 [mm/agno]')
ppChile.leer_xyvar_texto(ruta= 'PanualChile1979_2014.txt', namevar = 'precipitacion')


#SHAPE DE CHILE
shp = gpd.read_file('tempdir/regionesGEO.shp') 
ppChile.Plot(shape=shp,titulo='Precipitacion anual Chile',labelfig = 'mm/año',cmap = newcmp)




shp = gpd.read_file('tempdir/regionesGEO.shp') 





















figsize = (5,12)
fig, ax = plt.subplots(1, figsize=figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cax.set_title('Precipitacion [mm/año]')
ax.set_title('Precipitacion anual promedio Chile 1979-2014',fontsize=18,color='Blue')
ax.set_xlabel('Lon [°]',fontsize=15)
ax.set_ylabel('Lat [°]',fontsize=15)
##Limites del plot
#geodata = geo_df[geo_df['depth']>0]
min_lat = ppChile.geodf['latitude'].min()
max_lat = ppChile.geodf['latitude'].max()
min_lon = ppChile.geodf['longitude'].min()
max_lon = ppChile.geodf['longitude'].max()
#Geodataframe plot
top = plt.get_cmap('Blues', 8)
bottom = plt.get_cmap('BrBG', 50)
bottom = ListedColormap(bottom(np.linspace(0.2, 0.3, 2)))
top = ListedColormap(top(np.linspace(0.7, 1, 8)))
newcolors =  np.vstack((bottom(np.linspace(0, 1, 1)),top(np.linspace(0, 1, 8))))
newcmp = ListedColormap(newcolors, name='greens')
ppChile.geodf.plot(column= 'precipitacion',legend=True,markersize=1,marker='s',
            cmap=newcmp,edgecolor="none",ax=ax,cax=cax)
#shape de calibracion plot
shp.boundary.plot(ax=ax, color="black",linestyle='solid',
      linewidth=1,label='Chile')
ax.set_xlim([min_lon,max_lon])
ax.set_ylim([min_lat,max_lat])
##LEYENDA MANUAL
# where some data has already been plotted to ax
handles, labels = ax.get_legend_handles_labels()
# manually define a new patch 
#patch = mpatches.Patch(color='darkgoldenrod', label='Quebrada')
#patch2 = mpatches.Patch(color='blue', label='Deposito modelado')
# handles is a list, so append manual patch
#handles.append(patch)
#handles.append(patch2)
ax.legend(handles=handles,loc=4)
##Flecha Norte
x, y, arrow_length = 0.98, 0.99, 0.1
ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=10, headwidth=20),
            ha='center', va='center', fontsize=30,
            xycoords=ax.transAxes)
#guardar la figura
fig.savefig(os.getcwd() + '/' 'MapaChilePPpromedio.pdf',dpi=400)

        
    

#creacion de un archivo CSV
#df.to_csv (os.getcwd() + '/debris_data.csv', index = False, header=True)






#PAISES CERCANOS A CHILE
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

cmap = matplotlib.cm.get_cmap('cool')
cmap.set_under('red')

figsize = (5,12)
fig, ax = plt.subplots(1, figsize=figsize)
world.loc[world['continent']=='South America',:].plot(column= 'pop_est',legend=True,#markersize=1,marker='s',
            cmap=cmap,edgecolor="black", ax = ax,
            legend_kwds={'label': 'Poblacion',
                         'orientation': "horizontal"},
            vmin=10000)
world.boundary.plot(linewidth=0.3,color='black');



figsize = (5,12)
fig, ax = plt.subplots(1, figsize=figsize)
world.loc[world['continent']=='South America',:].plot(column= 'pop_est',legend=True,#markersize=1,marker='s',
            cmap=plt.cm.get_cmap('Blues', 10),edgecolor="black", ax = ax,
            legend_kwds={'label': 'Poblacion',
                         'orientation': "horizontal"},
            vmin=10000)
world.boundary.plot(ax=ax,linewidth=0.3,color='black');





countries = ['Chile','Argentina','Peru','Bolivia','Brazil']
idx_Chile = [ int(np.where(world['name'] == i )[0]) for i in countries  ] 

ShapeChile = world.iloc[ idx_Chile ,:]

ShapeChile.boundary.plot(linewidth=0.3,color='black');

fig , axes = plt.subplots( nrows= 1 , ncols=1,figsize=(6,10),
                          sharex=True, sharey=True)
ShapeChile.boundary.plot(ax = axes,linewidth=1,color='black');
axes.set_xlim([-80,-65])
axes.set_ylim([-55,-10])






















