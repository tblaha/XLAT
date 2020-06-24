import numpy  as np
import pandas as pd
import random as rd

def m2np(x):
    """
    Hack to change the pd.DataFrame of a measurement to an np.array something

    Parameters
    ----------
    x : list(strings)
        input objects of the measurements as read by pandas from the excel.

    Returns
    -------
    n : array(length(x))
        Stations having received the aircraft.
    ns : array(length(x))
        Station TOA in nanosec.
    R : array(length(x))
        Station reception RSSI.

    """
    a = np.array(list(map(int,list(map(float, str(x).replace("],[",",").replace("[","").replace("]","").split(","))) )))
    
    if a.size < 3:
        nparr = np.array([])
        return nparr,nparr,nparr
    elif a.size%3:
        exit("Input measurement list could not be processed properly.")
    else:
        nparr = a.reshape( int(a.size/3), 3 )
        return nparr[:,0],nparr[:,1],nparr[:,2]


def readMeasurements(filename, K = 0):
    """
    Read the measurements files  into a pd.DataFrame

    Parameters
    ----------
    filename : string
        The file to read.
    K : int, optional
        How many (randomly determined) rows to read. 0 if all. The default is 0.

    Returns
    -------
    MR : pd.DataFrame
        Measurement Frame:
        'id'|   't'       |'ac'|'lat'|'long'|'baroAlt'|'geoAlt'|  'M'      |    'm'   |    'n'      |   'ns'      | 'R' 
        'id'| server time |'ac'|'lat'|'long'|'baroAlt'|'geoAlt'| #stations | raw meas.| station ids | station TOA | RSSI
        
        everything starting from 'lat' may be empty or NaN depending on 
        measurment situation
    """
    
    # naive read
    MR = pd.read_csv(filename, delimiter=',')
    
    # rename columns
    MR = MR.rename(columns = {'timeAtServer':'t', 'aircraft':'ac', 'latitude':'lat', 'longitude':'long', 'baroAltitude':'baroAlt', 'geoAltitude':'geoAlt', 'numMeasurements':'M', 'measurements':'m'})
    # id (1 through 2,074,193) | server time | aircraft | lat | long | baroAlt | geoAlt | numMeasurements | measurements | reciever Nodes | nano seconds gps time | RSSI
    # id | t | ac | lat | long | baroAlt | geoAlt | M | m | n | ms | R
    
    # randomly select K lines or just use all
    Mrows = len(MR)
    if K > 0:
        MR = MR.loc[rd.sample(range(Mrows), K)]
    
    # apply a hack to unpack the m column into the last 3
    MR["n"], MR["ns"], MR["R"] = zip(*MR.m.apply(m2np))
    
    return MR


def readNodes(filename):
    """
    Read the station data into a pd.DataFrame
    

    Parameters
    ----------
    filename : string
        The file to read.

    Returns
    -------
    NR : pd.DataFrame
        contains location and type info for all stations
        |'n'|'lat'|'long'|'geoAlt'| (geoAlt in meters)
    """
    
    # naive read
    NR = pd.read_csv(filename, delimiter=',')
    
    #rename columns
    NR = NR.rename(columns = {'serial':'n', 'latitude':'lat', 'longitude':'long', 'height':'geoAlt'})
    # id (1 through 2,074,193) | server time | aircraft | lat | long | baroAlt | geoAlt | numMeasurements | measurements | reciever Nodes | nano seconds gps time | RSSI
    # id | t | ac | lat | long | baroAlt | geoAlt | M | m | n | ms | R
        
    return NR


def readSolutions(filename):
    """
    Read validation set solutions from different file

    Parameters
    ----------
    filename : string
        The file to read.

    Returns
    -------
    SR : pd.DataFrame
        contains location for some of the aircraft with missing locations
        |'id', 'lat', 'long', 'geoAlt'|
    """
    
    # naive read
    SR = pd.read_csv(filename, delimiter=',')
    
    # rename columns
    SR = SR.rename(columns = {'latitude':'lat', 'longitude':'long', 'geoAltitude':'geoAlt'})
    
    return SR

