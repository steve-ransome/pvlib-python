# USING MECHANISTIC PERFORMANCE/LOSS FACTORS MODELS MLFM With PVLIB
# Author : Steve Ransome SRCL mlfm_lib.py

py_ver = "200817T17" # version of this py file

#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import FuncFormatter
from pandas.plotting import register_matplotlib_converters
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import os
import sys
import pvlib # not used here yet

from scipy.optimize import curve_fit
from scipy import optimize
import math

# import ipywidgets as widgets # not used yet

#-------------------------------------------------------------------------------

# print information on loading
start = dt.datetime.now()
print ("mlfm_lib file loaded")
print ("start = ",start)
print ('py_ver =', py_ver)

#-------------------------------------------------------------------------------

# DETAILS OF PANDAS DATAFRAMES USED IN MLFM/LIB

#-------------------------------------------------------------------------------

# ref : DataFrame

# ''datasheet/reference module STC values and temperature coefficents''
#   note ff_ref = (imp_ref/isc_ref) * (vmp_ref/voc_ref)
#                                                                         column
# Contains the following keys:
#     * ``row``                : row number                      [integer]  #1
#     * ``site``               : site name                       [string]   #2
#     * ``module``             : module name                     [string]   #3
#     * ``id``                 : module id                       [string]   #4
#
#     * ``isc_ref``            : stc i_short_circuit             [A]        #5
#     * ``imp_ref``            : stc i_max_power                 [A]        #6
#     * ``vmp_ref``            : stc v_max_power                 [V]        #7
#     * ``voc_ref``            : stc v_open_circuit              [V]        #8
#
#     * ``alpha_isc_ref_norm`` : 1/isc * disc/dtmod              [1/K]      #9
#     * ``alpha_imp_ref_norm`` : 1/imp * dimp/dtmod              [1/K]      #10
#     * ``beta_vmp_ref_norm``  : 1/vmp * dvmp/dtmod              [1/K]      #11
#     * ``beta_voc_ref_norm``  : 1/voc * dvoc/dtmod              [1/K]      #12
#     * ``gamma_pmp_ref_norm`` : 1/pmp * dpmp/dtmod              [1/K]      #13
#
#     * ``active``             : is data available ?             [0 or 1]   #14
#     * ``comments``           : comments                        [string]   #15
#     * ``filename``           : filename                        [string]   #16

#-------------------------------------------------------------------------------

# meas : DataFrame

#  ''date,weather and raw measured module electrical values from IV curves''
#                                                                       columns
#                                                                    NREL 1-15
#                                                                 GANTNER 1-29
# Contains the following keys:
#     * ``date_time``          : iso 8601 'yyyy-mm-dThh:mm:ss'   [datetime] #1
#     * ``gti``                : poa global irradiance           [W/m^2]    #2
#     * ``wind_speed``         : wind speed                      [m/s]      #3
#     * ``temperature_air``    : ambient air temperature         [C]        #4
#     * ``temperature_module`` : module temperature              [C]        #5
#     * ``isc``                : short-circuit current un tcorr  [A]        #6
#     * ``rsc``                : -1/(dI/dV)|V=0 (R at isc)       [Ohms]     #7
#     * ``imp``                : max power point current         [A]        #8
#     * ``vmp``                : max power point voltage         [V]        #9
#     * ``roc``                : -1/(dI/dV)|I=0 (R at voc)       [Ohms]     #10
#     * ``voc``                : open circuit voltage un tcorr   [V]        #11
#     * ``pmp``                : power max power point           [W]        #12
#     * ``i_half_vmp``         : i @ v=vmp/2                     [A]        #13
#     * ``v_half_imp``         : v @ i=imp/2                     [V]        #14
#     * ``fraction_beam``      : beam fraction = 1 - dhi/ghi     [float]    #15
#
#     * ``ghi``                : global horizontal irradiance    [W/m^2]    #16
#     * ``dhi``                : diffuse horizontal irradiance   [W/m^2]    #17
#     * ``gni``                : global normal                   [W/m^2]    #18
#     * ``relative humididty`` : relative humidity 0 - 1         [float]    #19
#     * ``air_mass``           : air mass e.g. 1.5               [float]    #20
#     * ``solar_elevation``    : solar elevation angle 0 - 90    [degrees]  #21
#     * ``solar_azimuth``      : solar azimuth 0N - 90E - 360    [degrees]  #22
#     * ``ghi``                : global horizontal irradiance    [W/m^2]    #23
#     * ``xni``                : extraterrestrial normal irrad   [W/m^2]    #24
#     * ``xhi``                : extraterrestrial horiz irrad    [W/m^2]    #25
#     * ``clearness_h``        : clearness index horiz 0 - 1     [float]    #26
#     * ``clearness_n``        : clearness index normal 0 - 1    [float]    #27
#     * ``gti_refcell``        : poa irradiance cSi ref cell     [W/m^2]    #28
#     * ``gti_kg3``            : poa irradiance cSi kg3 glass    [W/m^2]    #29

#     * ``gti_kw_m2``          : converted poa global irradiance [kW/m^2]   #30

#-------------------------------------------------------------------------------

# norm : DataFrame

# ''date,weather and normalised (multiplicative) mlfm values derived
# from ref and meas''
# e.g.
#    norm['isc'] = meas['isc'] / ref['isc'] / meas['gti_kw_m2']
#    norm['voc'] = meas['voc'] / ref['voc']
# prdc is the product of 6 multiplicative mlfm factors scaled by 1/ff_ref
#    norm['prdc'] = 1/ff_ref * \
#        (norm['isc'] * norm['rsc'] * norm['ffi']) * \
#        (norm['ffv'] * norm['roc'] * norm['voc'])
#
#    = 1/ff_ref * norm['imp'] * norm ['vmp']
#
# Contains the following keys:
#     * ``date_time``          : iso 8601 'yyyy-mm-dThh:mm:ss'   [datetime] #1
#     * ``gti_kw_m2``          : poa global irradiance           [kW/m^2]   #2
#     * ``wind_speed``         : wind speed                      [m/s]      #3
#     * ``temperature_air``    : ambient air temperature         [C]        #4
#     * ``temperature_module`` : module temperature              [C]        #5
#
#     * ``isc``                : norm. no temp. corr. isc        [float]    #6
#     * ``rsc``                : norm. rsc                       [float]    #7
#     * ``ffi``                : norm. fill factor current       [float]    #8
#     * ``ffv``                : norm. fill factor voltage       [float]    #9
#     * ``roc``                : norm. roc                       [float]    #10
#     * ``voc``                : norm. no temp. corr. voc        [float]    #11
#     * ``imp``                : norm. imp                       [float]    #12
#     * ``vmp``                : norm. vmp                       [float]    #13
#     * ``prdc``               : no temp corr perf. ratio        [float]    #14
#
#     * ``icurve``             : current curvature factor        [float]    #15
#     * ``vcurve``             : voltage curvature factor        [float]    #16
#
#
#     * ``isc_tcorr``          : norm. temp. corr. isc           [float]    #17
#     * ``voc_tcorr``          : norm. temp. corr. voc           [float]    #18
#     * ``prdc_tcorr``         : temp. corr. perf. ratio         [float]    #19

#-------------------------------------------------------------------------------

#     References
#     ----------
#
#     The Loss Factors Model (LFM) and Mechanistic Performance Model (MPM)
#     together known as MLFM have been developed by SRCL and Gantner Instruments
#     (previously Oerlikon Solar and Tel Solar) since 2011 MLFM and 2017 MPM
#
#     .. [1] J. Sutterlueti(now Gantner Instruments) and  S. Ransome
#        '4AV.2.41 Characterising PV Modules under Outdoor Conditions:
#        What's Most Important for Energy Yield'
#        26th EU PVSEC 8 September 2011; Hamburg, Germany
#        http://www.steveransome.com/pubs/2011Hamburg_4AV2_41.pdf
#
#     .. [2] S. Ransome and J. Sutterlueti (Gantner Instruments)
#        'Checking the new IEC 61853.1-4 with high quality 3rd party data to
#        benchmark its practical relevance in energy yield prediction'
#        PVSC June 2019 [Chicago], USA
#        http://www.steveransome.com/PUBS/1906_PVSC46_Chicago_Ransome.pdf
#
#     .. [3] Steve Ransome (SRCL) and Juergen Sutterlueti (Gantner Instruments)
#        '5CV.4.35 Quantifying Long Term PV Performance and Degradation
#        under Real Outdoor and IEC 61853 Test Conditions
#        Using High Quality Module IV Measurements'
#        36th EU PVSEC Sep 2019 [Marseille]
#
#     .. [4] Steve Ransome (SRCL)
#        'How to use the Loss Factors and Mechanistic Performance Models
#        effectively with PVPMC/PVLIB'
#        [PVPMC] Webinar on PV Performance Modeling Methods, Aug 2020
#        https://pvpmc.sandia.gov/download/7879/
#
#     .. [5] Many more papers are available at www.steveransome.com

#-------------------------------------------------------------------------------

# REFERENCE CONDITIONS AND VALUES
# G=irradiance, T=Tmod, W=WindSpeed
#
# NAME  value       comment                    unit       PV_LIB name

T_STC =  25       # STC standard temperature   [C]        temperature_ref
G_STC =  1        # STC standard irradiance    [kW/m^2]
G_LIC =  0.2      # LIC low irradiance         [kW/m^2]
G_NOCT = 0.8      # NOCT irradiance            [kW/m^2]
G_MAX =  1.4      # for max plot limit         [kW/m^2]
G_MIN =  0.01     # for min plot limit         [kW/m^2]
T_MAX =  80       # for max plot limit         [C]
T_MIN =  -20      # for max plot limit         [C]

#-------------------------------------------------------------------------------

# STANDARDISED COLOURS (*_col) FOR MLFM PLOTS
#
#             python name    rrr ggg bbb  comment
#
gi_col     = 'darkgreen'   # Irradiance   irradiance
tmod_col   = 'red'         # 255   0   0  temperature_module
tamb_col   = 'beige'       # 245 245 220  temperature_ambient
ws_col     = 'grey'        # 127 127 127  wind_speed
#
isc_col    = 'purple'      # 128   0 128
rsc_col    = 'orange'      # 255 165   0
ffi_col    = 'lightgreen'  # 144 238 144
ffv_col    = 'cyan'        #   0 255 255
imp_col    = 'green'       #   0 255   0
vmp_col    = 'blue'        #   0   0 255
roc_col    = 'pink'        # 255 192 203
voc_col    = 'sienna'      # 160  82  45
#
icurve_col = 'yellowgreen' # 178 217 100
vcurve_col = 'turquoise'   #  67 224 208
prdc_col   = 'black'       #   0   0   0
#
kTh_col    = 'grey'        # 127 127 127 # clearness kTh

#-------------------------------------------------------------------------------

# date formatting

years      = mdates.YearLocator()   # every year
months     = mdates.MonthLocator()  # every month
days       = mdates.DayLocator()  # every month
hours      = mdates.HourLocator()  # every hour
years_fmt  = mdates.DateFormatter('%Y')
months_fmt = mdates.DateFormatter('%Y-%m')
days_fmt   = mdates.DateFormatter('%d-%m')

register_matplotlib_converters()

#-------------------------------------------------------------------------------

# default figure size and definitions for plots

plt.rcParams['figure.figsize'] = (16, 10)
ms = 20  # marker_size
mk = 'o' # marker
mpl.rcParams.update({
        'font.size': 12, 
        'font.weight':'bold', 
        'text.color':'darkblue'
    })

#-------------------------------------------------------------------------------

# list names (and numbers) of mlfm variables

mlfm_param_names = [
           'prdc',         # 0
           'isc',          # 1
           'rsc',          # 2
           'ffi' ,         # 3
           'ffv',          # 4
           'roc',          # 5
           'voc',          # 6
           'imp',          # 7
           'vmp',          # 8
           'icurve',       # 9
           'vcurve',       # 10
           'prdc_tcorr',   # 11
           'isc_tcorr',    # 12
           'voc_tcorr'     # 13
          ]

#-------------------------------------------------------------------------------
#lib_level =1 #
#-------------------------------------------------------------------------------

def get_meas_filename(mlfm_file_names, show_data = False):
    ''' get user choice of measured data files'''

    # Parameters
    # ----------
    # mlfm_file_names : list
    #     list of measurement files meass\*.csv
    #
    # show_data : boolean
    #     show data from meas dataframes
    #
    # Returns
    # -------
    # meas_filename : string
    #    neme of file selected
    #
    # meas_data
    #    meas dataframe
    #
    # ref_sel : dataframe
    #    row of ref module data
    #
    # mod_row : integer
    #    row number of reference module selected
    #
    # Notes
    # -----
    # could automate from a directory listing but that causes
    # problems when non meas type files are listed too

    print ('\n show user row number and file names')
    for i, filename in enumerate(mlfm_file_names):
        print (i, '=', filename)

    # get user choice of file name
    qty_file_names = (len(mlfm_file_names) - 1)
    print ()
    while True:
        module_select = input_integer(
            '\n ## Choose module (0) to (' +
            str(qty_file_names) + ') : ')

        print
        if ((module_select >= 0) and
            (module_select <= qty_file_names)):
            break
        else:
            print ('Select module (0) to (' +
                   str(qty_file_names) + ') ')

    meas_filename = mlfm_file_names[module_select]

    # find module row in ref
    mod_row = module_select
    print ('file selected = ' + meas_filename)

    # read measured file data
    meas_data = pd.read_csv('meas\\'+meas_filename, skiprows=0)

    # perform a simple data sanity check
    # further checks are done later on
    meas_data.dropna()
    meas_data = meas_data[meas_data['gti'] > 1]
    meas_data = meas_data[meas_data['pmp'] > 0]

    # translate irradiance W/m2 --> kW/m2
    # as it's easier for normalised data (as G_STC = 1)
    # could rename column and divide by 1000?
    meas_data['gti_kw_m2'] = meas_data['gti']/1000

    if show_data == True:
        meas_data.describe()

    # load reference data for modules from ref file
    ref_filename = 'ref\\ref.csv' # ref file name
    ref_data = pd.read_csv(ref_filename, skiprows = 0)

    # select the row that matches module_select
    # presently this data has 'meas_row = ref_row' but it would be better to
    # search for module name
    # if the user adds meas data then they must add the relevant ref data too
    # if the wrong row type is used then the normalised data will appear wrong
    # for example if a cSi module has ref['isc']=10A and a CdTe module from
    # ref is used with ref['isc']=2A then the norm['isc'] becomes ~5 not ~1
    
    ref_sel = ref_data[ref_data.row == module_select ].copy()

    # show the STC module data values in ref, check you have the correct row
    # ref_sel.head(1)
    # print ('ref_sel = ',ref_sel)
    
    return meas_filename, meas_data, ref_sel, mod_row #, ref_data

#-------------------------------------------------------------------------------

def plot_mlfm_xaxis(xaxis_name, dframe, dfilename, dfilter,
                show_lfm_vals = 1, save_graphs = True,
                ymax = 1.1, ymin = 0.7, xmax = 1.2, xmin = 0,
               ):
    ''' 
    Plot many normalised Loss Factors Model parameters (e.g. norm['rsc'] .. 
    norm['voc_tcorr']) on the y1 axis vs. user selected xaxis e.g.  
    'gti_kw_m2', 'temperature_module' or 'date_time' 
    '''

    # Parameters
    # ----------
    # xaxis_name : string
    #     'gti_kw_m2'(irradiance), 'temperature_module' or other from dataframe
    #
    # dframe : dframe
    #     dframe of mlfm values (e.g. prdc, isc, rsc, .. voc).
    #
    # dfilename : filename
    #     'institute_site_technology_moduleid_measfreq_duration_numrows' e.g.
    #     'nrel_cocoa_mSi_0188_15m_2011_400.csv'
    #
    # show_lfm_vals : integer
    #     1=roc-voc, 2= ++isc,prdc, 3= ++curves, 4= ++tcorr
    #
    # dfilter : string
    #     User chosen to Identify 'sanity check filter variables and limits'
    #      e.g. '0p4Gi0p8 --> 0.4 < gi < 0.8
    #     note : this is added to png graph file name
    #          do not use illegal characters -->  ( < > : " / \ | ? * ) #
    #
    # save_graphs : boolean
    #     save graph as a .png file
    #
    # optional
    #
    # ymax : float
    #     maximum yscale usually ~ 1.1
    #
    # ymin : float
    #     minimum yscale usually 0 or ~ 0.7
    #
    # xmax : float
    #     maximum xscale usually ~ 1.2
    #
    # xmin : float
    #     minimum xscale usually ~ 0.
    #
    # Returns
    # -------
    #     scatter plot of mlfm parameters vs. xaxis type 'gti_kw_m2'
    #     ,'temperature_module','date_time'.

    # Setup graph
    fig, ax1 = plt.subplots()
    ax1.grid( color='grey', linestyle='-', linewidth=1)

    graph_title =  'mlfmx__' + dfilename[:len(dfilename)-4] + \
            '__x=' + xaxis_name + '_show_all=' + str(show_lfm_vals)

    ax1.set_title(graph_title)

    # y1-axis
    ax1.set_ylabel('mlfm values')
    ax1.set_ylim(ymin, ymax)
    ax1.yaxis.set_ticks(np.arange(ymin, ymax, .1))
    ax1.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
    ax1.axhline(y = 1, color = 'grey', linewidth = 3) # show 100% line

    # x-axis
    xdata = dframe[xaxis_name]

    ax1.set_xlabel(xaxis_name)

    if xaxis_name == 'gti_kw_m2':
        # irradiance
        ax1.set_xlim(0, G_MAX) # x scale
        ax1.xaxis.set_ticks(np.arange(0, G_MAX, 0.2)) # x tick
        # show important values
        ax1.axvline(x = G_LIC, c = 'darkgreen', linewidth = 3) # 0.2kW/m^2 = LIC
        ax1.axvline(x = G_STC, c = 'green', linewidth = 3) # 1.0kW/m^2 = STC

    elif xaxis_name == 'temperature_module':
        # temperature_module
        ax1.set_xlim(0, T_MAX ) # optional x scale to T_MAX)
        # show important value
        ax1.xaxis.set_ticks(np.arange(0, T_MAX, 5))
        ax1.axvline(x = T_STC, c = 'red', linewidth = 3)

    else:
        ax1.set_xlim(xmin, xmax) # optional x scale
        print ('Warning : undefined xaxis_type in plot_mlfm_xaxis()')

    # scatter plot the most common mlfm parameters
    #               xaxis                  yaxis         size   colour   marker

    if show_lfm_vals >= 1:
        ax1.scatter(xdata, dframe['rsc'],        ms,   rsc_col, mk, label = 'rsc')
        ax1.scatter(xdata, dframe['ffi'],        ms,   ffi_col, mk, label = 'ffi')
        ax1.scatter(xdata, dframe['ffv'],        ms,   ffv_col, mk, label = 'ffv')
        ax1.scatter(xdata, dframe['roc'],        ms,   roc_col, mk, label = 'roc')
        ax1.scatter(xdata, dframe['voc_tcorr'],  ms,   voc_col, mk, label = 'voc_tcorr')
    if show_lfm_vals >= 2:
        # more scattered params (epending on soil, aoi, spectrum)
        ax1.scatter(xdata, dframe['isc_tcorr'],  ms/2, isc_col, 's',  label = 'isc_tcorr')
        ax1.scatter(xdata, dframe['prdc_tcorr'], ms/2, prdc_col, '+', label = 'prdc_tcorr')
    if show_lfm_vals >= 3:
        # show more derived parameters
        ax1.scatter(xdata, dframe['icurve'],     ms/4, icurve_col,  mk, label = 'icurve')
        ax1.scatter(xdata, dframe['vcurve'],     ms/4, vcurve_col,  mk, label = 'vcurve')
    if show_lfm_vals >= 4:
        ax1.scatter(xdata, dframe['imp'],        ms,   imp_col, mk, label = 'imp')
        ax1.scatter(xdata, dframe['vmp'],        ms,   vmp_col, mk, label = 'vmp')

    #y2 axis
    # show weather data Gi, Tair and Tmod # and ws?
    ax2 = ax1.twinx()
    ax2.set_ylabel('Gi (kW/m^2); Tmod, Tair (C/100)')
    ax2.set_ylim(0, 4)             # set wide so doesn't overlap lfm params
    ax2.scatter(xdata, dframe['gti_kw_m2'],               ms,   gi_col,    mk, label = 'gi')
    ax2.scatter(xdata, dframe['temperature_module']/100,  ms,   tmod_col,  mk, label = 'tmod')
    ax2.scatter(xdata, dframe['temperature_air']/100,     ms,   tamb_col,  mk, label = 'tamb')

    ax2.legend(frameon=True, loc='upper right',  shadow=True)

    ax1.legend(frameon=True, loc='lower right',  shadow=True)

    if save_graphs == True:
        plt.savefig('graphs//' + graph_title + '_' + dfilter + '.png')

    plt.show()

#-------------------------------------------------------------------------------    
    
def plot_mlfm_xaxis_caxis(yaxis_name, xaxis_name, caxis_name,
                          dframe, dfilename, dfilter,
                          save_graphs = True,
                          ymax = 1.1, ymin = 0.7, xmax = 1.2, xmin = 0,
                         ):
    '''
    Plot one normalised Loss Factors Model parameter (e.g. nrsc, nvoc)
    on the y axis vs. user selected xaxis e.g. 'gti_kw_m2', 'temperature_module',
    'date_time'  with coloured dots and cbar for a different parameter caxis
    '''

    # Parameters
    # ----------
    # yaxis_name : string
    #     mlfm parameter from  dframe e.g. 'prdc', 'rsc', 'voc' ...
    #
    # xaxis_name : string
    #     'gti_kw_m2'(irradiance), 'temperature_module' or other column from dataframe
    #
    # caxis_name : string
    #     'temperature_module', 'gti_kw_m2'(irradiance)  or other column from dataframe
    #
    # dframe : dataframe
    #     dataframe of normalised values (e.g. 'isc', 'rsc', .. 'voc').
    #
    # dfilename : string
    #     'institute_site_technology_moduleid_measfreq_duration_numrows' e.g.
    #     'nrel_cocoa_mSi_0188_15m_2011_400.csv'
    #
    # dfilter : string
    #     User chosen to Identify 'sanity check filter variables and limits'
    #      e.g. '0p4Gi0p8 --> 0.4 < gi < 0.8
    #     note : this is added to png graph file name
    #          do not use illegal characters -->  ( < > : " / \ | ? * )
    #
    # optional
    #
    # ymax : float
    #     maximum yscale usually ~ 1.1
    #
    # ymin : float
    #     minimum yscale usually 0 or ~ 0.7
    #
    # xmax : float
    #     maximum xscale usually ~ 1.2
    #
    # ymin : float
    #     minimum xscale usually 0
    #
    # Returns
    # -------
    #     scatter plot
    #     y) mlfm parameters vs.
    #     x) xaxis 'gti_kw_m2', 'temperature_module', 'date_time'
    #     c) coloured by 'temperature_module' or 'gti_kw_m2'
    #

    # define y, x and colour axes data
    ydata = dframe[yaxis_name]
    xdata = dframe[xaxis_name]
    cdata = dframe[caxis_name]

    # setup
    fig, ax1 = plt.subplots()

    # plot a grey grid if not a datetime x axis
    if xaxis_name != 'date_time':
        ax1.grid( color='grey', linestyle='-', linewidth=1)

    # title
    graph_title = 'mlfmxc__' + dfilename[:len(dfilename)-4] + '__y' + \
        yaxis_name + '_x' + xaxis_name + '_c' + caxis_name
    ax1.set_title(graph_title)

    # y1-axis
    ax1.set_ylabel(yaxis_name)
    ax1.set_ylim(ymin, ymax)
    ax1.yaxis.set_ticks(np.arange(ymin, ymax, 0.1))

    ax1.axhline(y=1) # show nominal 1 = 100%
    ax1.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))

    # x-axis
    ax1.set_xlabel(xaxis_name)
    ax1.set_xlim(xmin, xmax)
    ax1.axvline(x = 1, c='grey')

    if xaxis_name == 'gti_kw_m2':
        ax1.axvline(x = G_STC, c = 'grey') # 1.0 kW/m^2 STC
        ax1.axvline(x = G_LIC, c = 'grey') # 0.2 kW/m^2 LIC

    elif xaxis_name == 'temperature_module':
        ax1.set_xlim(0, T_MAX)
        ax1.xaxis.set_ticks(np.arange(0, T_MAX, 10)) # optional x tick
        ax1.axvline(x = T_STC, c='grey') # = 25C STC

    elif xaxis_name == 'voc_tcorr':
        ax1.axvline(x = 1, c = 'grey') # 1.0 kW/m^2 STC

    else:
        ax1.set_xlim(xmin, xmax) # optional x scale
        print ('Warning: undefined xaxis_type in plot_mlfm_xaxis_caxis()')

    # plot and add colour map
    sc = ax1.scatter(
                     x = xdata,
                     y = ydata,
                     c = cdata,
                     alpha = 1,      # 1=opaque
                     cmap = mpl.cm.gist_rainbow,
                     s = ms,
                    )

    #ax1.legend(frameon=True, loc='lower right', shadow=True)

    # add colour bar but shrink to 75%
    fig.colorbar(sc, label=caxis_name, shrink=0.75)

    if save_graphs == True:
        plt.savefig('graphs//' + graph_title + '_' + dfilter + '.png')

    plt.show()

#-------------------------------------------------------------------------------

def mlfm_meas_to_norm (ref, meas, mod_row):
    ''' converts measured isc,rsc ... voc data (V,A,W,Ohms) to 
    normalised values (~1) '''

    # Parameters
    # ----------
    # ref : DataFrame
    #     datasheet/reference module electrical values 
    #     isc, imp ... voc (V, A, W ...)
    #
    # meas : DataFrame
    #     measured module electrical values from iv curve 
    #     isc,rsc ... voc (V, A, W, Ohmms ...)
    #
    # mod_row : integer
    #     selection row of module from reference datasheet to get
    #     the relevant isc, voc etc.
    #

    # create an empty dataframe 'norm' to contain the normalised mlfm data
    # could later add imp_tcorr and vmp_tcorr if needed
    #                                       comment
    norm = pd.DataFrame( columns = [
        'date_time',                      # weather data
        'gti_kw_m2',                      # ''
        'temperature_air',                # ''
        'temperature_module',             # ''
        'wind_speed',                     # ''
        'isc',                            # essential 6 mlfm parameters
        'rsc',                            # ''
        'ffi',                            # ''
        'ffv',                            # ''
        'roc',                            # ''
        'voc',                            # ''
        'icurve',                         # curvature values
        'vcurve',                         # ''
        'prdc',                           # optional temperature corrected
        'isc_tcorr',                      # ''
        'voc_tcorr',                      # ''
        'prdc_tcorr',                     # optional temperature corrected
        ])                  # datetime

    # copy weather data to norm for simplicity
    norm =  meas[['date_time',
                  'gti_kw_m2',
                  'wind_speed',
                  'temperature_air',
                  'temperature_module',
                ]].copy()

    # get stc reference values for selected module from ref
    isc_ref   = ref['isc_ref'][mod_row]
    imp_ref   = ref['imp_ref'][mod_row]
    vmp_ref   = ref['vmp_ref'][mod_row]
    voc_ref   = ref['voc_ref'][mod_row]
    pmp_ref = imp_ref * vmp_ref
    aisc_ref  = ref['alpha_isc_ref_norm'][mod_row]
    bvoc_ref  = ref['beta_voc_ref_norm'][mod_row]
    gpmp_ref  = ref['gamma_pmp_ref_norm'][mod_row]

    # calculate intermediate values (ir,vr) =
    #    'intercept of tangents to rsc@isc and roc@voc'
    ir = ((meas['isc'] * meas['rsc'] - meas['voc']) / 
              (meas['rsc'] - meas['roc']))

    vr = (meas['rsc'] * (meas['voc'] - meas['isc'] * meas['roc']) / 
              (meas['rsc'] - meas['roc']))

    # calculate normalised mlfm values
    norm['isc']   = meas['isc']  / (meas['gti_kw_m2'] * isc_ref)

    norm['rsc']   = ir / meas['isc']
    norm['ffi']   = meas['imp'] / ir
    norm['ffv']   = meas['vmp'] / vr
    norm['roc']   = vr / meas['voc']
    norm['voc']   = meas['voc'] / voc_ref

    norm['imp']   = meas['imp']  / (meas['gti_kw_m2'] * isc_ref )
    norm['vmp']   = meas['vmp']  / (voc_ref )

    # calc dc performance ratio - usually '0.5 < prdc < 1.05'
    norm['prdc'] = ((meas['imp'] * meas['vmp']) /
                        (pmp_ref * meas['gti_kw_m2'])) ###

    # calculate curvature parameters ic, vc to identify curvature/steps
    # icurve = 'I@Vmp/2 measured value' /
    #            'expected from extrapolating isc,rsc'
    # indicates if shading or mismatch v < vmp if ic not ~1.00
    norm['icurve'] = meas['i_half_vmp'] / \
        (meas['isc']-meas['vmp'] / (2 * meas['rsc']))

    # vcurve  = 'V@Imp/2 measured value' /
    #            'expected from extrapolating voc,roc'
    # shows rollover/other problems v > vmp if vc not ~1.00
    norm['vcurve'] = meas['v_half_imp'] / \
        (meas['voc'] - meas['imp'] / 2 * meas['roc'])

    # TEMPERATURE CORRECTIONS use suffix '*_tcorr'
    temp_delta = (norm['temperature_module'] - T_STC) # delta tmod from 25C

    norm['isc_tcorr'] = norm['isc'] * (1 - aisc_ref * temp_delta)

    norm['voc_tcorr'] = norm['voc'] * (1 - bvoc_ref * temp_delta)

    norm['prdc_tcorr'] = norm['prdc'] * (1 - gpmp_ref * temp_delta)

    return norm

#-------------------------------------------------------------------------------

def input_integer(message):
    '''force input as integer, not float, string, nan etc.'''

    while True:
        try:
            userInput = int(input(message))
        except ValueError:
            print('Warning, Not an integer! Try again.')
            continue
        else:
            return userInput
        break

