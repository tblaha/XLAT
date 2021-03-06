from .helper import SP2CART, C0

import os
import json

import numpy  as np
import numpy.linalg as la

import pandas as pd


def importData(use_pickle, use_file):
    if use_file == -1:
        path = "../Data/competition_evaluation"
        results_path = path
        fnameMR = "round1_competition.csv"
        fnameSR = "round1_sample_empty.csv"
    elif use_file > 0:
        path = "../Data/training_dataset/training_"+str(use_file)+"_category_1"
        results_path = path+"_result"
        fnameMR = "training_"+str(use_file)+"_category_1.csv"
        fnameSR = "training_"+str(use_file)+"_category_1_result.csv"

    # read csv files
    if os.path.isfile("./MR.pkl") and use_pickle:
        MR = pd.read_pickle("./MR.pkl")
    else:
        MR = readMeasurements(path+"/"+fnameMR)
        MR.to_pickle("./MR.pkl")

    if os.path.isfile("./NR.pkl") and use_pickle:
        NR = pd.read_pickle("./NR.pkl")
    else:
        NR = readNodes(path+"/sensors.csv")
        NR.to_pickle("./NR.pkl")

    if os.path.isfile("./SR.pkl") and use_pickle:
        SR = pd.read_pickle("./SR.pkl")
    else:
        SR = readSolutions(results_path+"/"+fnameSR)
        SR.to_pickle("./SR.pkl")

    return MR, NR, SR


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
    a = np.array(list(map(int, list(map(float, str(x).
        replace("],[", ",").replace("[", "").replace("]", "").split(","))
        ))))

    if a.size < 3:
        nparr = np.array([])
        return nparr, nparr, nparr
    elif a.size % 3:
        exit("Input measurement list could not be processed properly.")
    else:
        nparr = a.reshape(int(a.size / 3), 3)
        return nparr[:, 0], nparr[:, 1], nparr[:, 2]


def readMeasurements(filename):
    """
    Read the measurements files  into a pd.DataFrame

    Parameters
    ----------
    filename : string
        The file to read.

    Returns
    -------
    MR : pd.DataFrame
        Measurement Frame:
        'id'|   't'       |'ac'|'lat'|'long'|'baroAlt'|'geoAlt'|  'M'
            |    'm'   |    'n'      |   'ns'      | 'R'
        'id'| server time |'ac'|'lat'|'long'|'baroAlt'|'geoAlt'| #stations
            | raw meas.| station ids | station TOA | RSSI

        everything starting from 'lat' may be empty or NaN depending on
        measurment situation
    """

    # naive read
    MR = pd.read_csv(filename, delimiter=',', index_col=0)

    # rename columns
    MR = MR.rename(columns={'timeAtServer': 't',
                            'aircraft': 'ac',
                            'latitude': 'lat',
                            'longitude': 'long',
                            'baroAltitude': 'baroAlt',
                            'geoAltitude': 'geoAlt',
                            'numMeasurements': 'M',
                            'measurements': 'm',
                            })
    # id (1 through 2,074,193) | server time | aircraft | lat | long
    # | baroAlt | geoAlt | numMeasurements | measurements | reciever Nodes
    # | nano seconds gps time | RSSI
    # id | t | ac | lat | long | baroAlt | geoAlt | M | m | n | ms | R

    # apply a hack to unpack the m column into the last 3
    MR["n"], MR["ns"], MR["R"] = zip(*MR.m.apply(
        lambda x: zip(*json.JSONDecoder().decode(str(x)))
        ))

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
    NR = pd.read_csv(filename, delimiter=',', index_col=0)

    # rename columns
    NR = NR.rename(columns={'latitude': 'lat',
                            'longitude': 'long',
                            'height': 'geoAlt',
                            })
    # id (1 through 2,074,193) | server time | aircraft | lat | long
    # | baroAlt | geoAlt | numMeasurements | measurements | reciever Nodes
    # | nano seconds gps time | RSSI
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
    SR = pd.read_csv(filename, delimiter=',', index_col=0)

    # rename columns
    SR = SR.rename(columns={'latitude': 'lat',
                            'longitude': 'long',
                            'geoAltitude': 'geoAlt'
                            })

    return SR


def segmentData(MR, use_SR, SR=None, K=0, p=0):
    """


    Parameters
    ----------
    MR : TYPE
        DESCRIPTION.
    useSR : TYPE
        DESCRIPTION.
    SR : TYPE, optional
        DESCRIPTION. The default is None.
    K : TYPE, optional
        DESCRIPTION. The default is 0.
    p : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    """
    if use_SR:
        TRA = MR  # values that are in SR are already nan in MR
        VAL = SR
    else:
        # find datapoints with ground truth inside training set
        MR_GT = MR.loc[~np.isnan(MR.lat)]
        # id_GT = MR[~np.isnan(MR.lat)].id # where we have ground truth from
        # training set
        Mrows_GT = len(MR_GT)

        # get training and validation sets
        # tra_idx, val_idx = sklms.train_test_split(MR_GT.index,\
        #                                           test_size=p*K/Mrows_GT, \
        #                                           train_size=K/Mrows_GT)
        if not K:
            K = Mrows_GT

        tra_idx = np.random.choice(MR_GT.index,
                                   size=K,
                                   replace=False
                                   )
        val_idx = np.random.choice(tra_idx,
                                   size=int(K * p),
                                   replace=False
                                   )

        TRA = MR_GT.loc[tra_idx, :].copy()
        VAL = MR_GT.loc[val_idx, ['lat', 'long', 'geoAlt']].copy()

        # NaN out the VAL lines in the training set:
        TRA.loc[val_idx, ['lat', 'long', 'geoAlt']] = np.nan

    TRA = TRA.sort_index(ascending=True)
    VAL = VAL.sort_index(ascending=True)

    return TRA, VAL


def segmentDataByAC(MR, K, p):
    """


    Parameters
    ----------
    MR : TYPE
        DESCRIPTION.
    useSR : TYPE
        DESCRIPTION.
    SR : TYPE, optional
        DESCRIPTION. The default is None.
    K : TYPE, optional
        DESCRIPTION. The default is 0.
    p : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    """

    # find datapoints with ground truth inside training set
    MR_GT = MR.loc[~np.isnan(MR.lat)]
    Mrows_GT = len(MR_GT)

    # get aircraft id's
    ac_ids = np.unique(MR_GT['ac'])
    np.random.shuffle(ac_ids)

    chosen_ids_tra = []
    chosen_ids_val = []
    val_complete = False
    cnt = 0
    for idx, ac in enumerate(ac_ids):
        # ids of the current ac
        cur_pnts = MR_GT.index[MR_GT['ac'] == ac]
        pnts = len(cur_pnts)
        
        cnt += pnts
        if (cnt <= K*p):
            # val set won't be complete with current ac
            chosen_ids_val.append(list(cur_pnts))
        elif not val_complete:
            # val set complete with (part of) current ac
            chosen_ids_val.append(list(cur_pnts[:int(pnts - (cnt - K*p))
                                                ]))
            val_complete = True

        if (cnt <= K):
            # tra set not yet complete
            chosen_ids_tra.append(list(cur_pnts))
        else:
            # val set complete with (part of) current ac
            chosen_ids_tra.append(list(cur_pnts[:int(pnts - (cnt - K))
                                                ]))
            break

    # flatify lists
    chosen_ids_tra_flat = [it for sl in chosen_ids_tra for it in sl]
    chosen_ids_val_flat = [it for sl in chosen_ids_val for it in sl]

    # copy result to TRA and VAL arrays
    TRA = MR_GT.loc[chosen_ids_tra_flat, :].copy()
    VAL = MR_GT.loc[chosen_ids_val_flat, ['lat', 'long', 'geoAlt']].copy()

    # NaN out the VAL lines in the training set:
    TRA.loc[chosen_ids_val_flat, ['lat', 'long', 'geoAlt']] = np.nan

    # sort by index
    TRA = TRA.sort_index()
    VAL = VAL.sort_index()

    return TRA, VAL

# ###################
# ## faking stuff ###
# ###################


def insertFakeStations(NR, sph_pos):
    """


    Parameters
    ----------
    NR : TYPE
        DESCRIPTION.
    sph_pos : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    # add fake stations
    idx_start = NR.index[-1] + 1
    for i in range(len(sph_pos)):
        NR.loc[idx_start + i] = {"lat": sph_pos[i, 0],
                                 "long": sph_pos[i, 1],
                                 "geoAlt": sph_pos[i, 2],
                                 "type": "Test "+str(i),
                                 }

    return NR, np.arange(idx_start, idx_start + len(sph_pos))


def insertFakePlanes(MR, NR, sph_pos, n, noise_amp=0):
    """


    Parameters
    ----------
    MR : TYPE
        DESCRIPTION.
    NR : TYPE
        DESCRIPTION.
    sph_pos : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.
    noise_amp : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    global SP2CART, C0

    start_idx = MR.index[-1] + 1

    # convert nodes to cartesian for calculating TOAs
    for idx, nitem in enumerate(n):
        MR.loc[start_idx + idx] = ""
    
        # get associated stations and convert to cartesian
        node_sph = np.array(NR.loc[nitem, ["lat", "long", "geoAlt"]])
        node_cart = SP2CART(node_sph)

        # convert plane location to cartesian
        cart_loc = SP2CART(sph_pos[idx]).T

        # find nanoseconds TOA (in this case equal to TOT, since no offset)
        ns = la.norm(node_cart - cart_loc, axis=1) / C0 * 1e9

        # add noise
        ns = np.vectorize(int)(
            ns + (np.random.random(len(nitem)) - 0.5) * 2 * noise_amp
            )

        # add fake plane
        MR.loc[start_idx + idx] = {
            "t": 1e3 - 1,
            "ac": int(1e4 - 1),
            "lat": sph_pos[idx, 0],
            "long": sph_pos[idx, 1],
            "baroAlt": sph_pos[idx, 2],
            "geoAlt": sph_pos[idx, 2],
            "M": int(len(nitem)),
            "m": "N/A",
            "n": nitem,
            "ns": tuple(ns),
            "R": tuple(ns*0),
            }

    return MR, np.arange(start_idx, start_idx + idx + 1)