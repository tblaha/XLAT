# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:26:06 2020

@author: Till
"""

import numpy as np
from numpy import linalg as la
from geopy.distance import great_circle as gc

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from constants import R0, X, Y, Z


def twoErrorCalc(x, z, RMSEnorm=2):
    """
    2 dimensional RMSE using great circle distance on the ground truth height

    Parameters
    ----------
    x : pd.DataFrame
        Generated solution dataset (validation set).
    z : pd.DataFrame
        Ground truth solution dataset (validation set).
    RMSEnorm : scalar, optional
        RMSE root to use. The default is 2.

    Returns
    -------
    e : scalar
        RMSE.

    """
    global R0, X, Y, Z

    # find the common indices (computed into x and preset in validation set z)
    sol_idx_bool = np.in1d(x.index, z.index)
    N = len(z.index)

    # get lat and longs and ground truth geo height
    lat_x, long_x = \
        np.array(x.loc[sol_idx_bool, ['lat', 'long']]).T
    lat_z, long_z, h_z = \
        np.array(z.loc[sol_idx_bool, ['lat', 'long', 'geoAlt']]).T

    # compute great circle distances ("2d" error) between guess and truth
    norm_vec = np.zeros(N)
    for i in range(N):
        try:
            norm_vec[i] = gc((lat_x[i], long_x[i]),
                             (lat_z[i], long_z[i])).meters\
                            * (R0+h_z[i])/R0
            if np.isnan(norm_vec[i]) or norm_vec[i] > 2.5e5:
                # if np.isnan(norm_vec[0, i]):
                norm_vec[i] = 0
                N = N - 1
        except ValueError:
            norm_vec[i] = 0
            N = N - 1

    # RMSE error sum
    e = (np.sum(norm_vec**RMSEnorm)/N)**(1/RMSEnorm)

    return e, norm_vec


def threeErrorCalc(x, z, RMSEnorm=2, pnorm=2):
    """
    3 dimensional RMSE using pnorm on cartesian coordinates.
    
    Parameters
    ----------
    x : pd.DataFrame
        Generated solution dataset (validation set).
    z : pd.DataFrame
        Ground truth solution dataset (validation set).
    RMSEnorm : scalar, optional
        RMSE root to use. The default is 2.
    pnorm : scalar, optional
        pnorm for the cartesian distance calculation. The default is 2.

    Returns
    -------
    e : scalar
        RMSE.

    """
    global R0, X, Y, Z

    # find the common indices (computed into x and preset in validation set z)
    sol_idx_bool = np.in1d(x.index, z.index)
    N = len(z.index)

    # get lat and longs and ground truth geo height
    lat_x, long_x, h_x = \
        np.array(x.loc[sol_idx_bool, ['lat', 'long', 'geoAlt']]).T

    lat_z, long_z, h_z = \
        np.array(z.loc[sol_idx_bool, ['lat', 'long', 'geoAlt']]).T

    # convert to cartesian
    cart_x = [X(lat_x, long_x, h_x),
              Y(lat_x, long_x, h_x),
              Z(lat_x, long_x, h_x)
              ]
    cart_z = [X(lat_z, long_z, h_z),
              Y(lat_z, long_z, h_z),
              Z(lat_z, long_z, h_z)
              ]

    # compute great circle distances ("2d" error) between guess and truth
    norm_vec = la.norm(np.array(cart_z) - np.array(cart_x), pnorm, 0)
    # broken = (np.isnan(norm_vec)) | (norm_vec > 2.5e5)
    broken = (np.isnan(norm_vec) | (norm_vec > 1e6))
    norm_vec[broken] = 0
    N = N - sum(broken)

    # RMSE error sum
    e = (np.sum(norm_vec**RMSEnorm)/N)**(1/RMSEnorm)

    return e, norm_vec


def writeSolutions(filename, z):
    """
    write solution DataFrame to csv.

    Parameters
    ----------
    filename : string
        DESCRIPTION.
    z : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    int
        DESCRIPTION.

    """
    zz = z.copy()
    zz.columns = ['latitude', 'longitude', 'geoAltitude']
    zz.to_csv(filename, index=True, index_label='id',
              na_rep='NaN')  # not-a-number string

    return 0


class PlanePlot():
    """
    Handles plotting of planes, tracks, stations and other points on a
    political world map
    """

    def __init__(self):
        """
        Generate initial plot window with the political map
        """
        self.fig = plt.figure(figsize=(15, 8))
        # self.ax  = self.fig.add_subplot(1,1,1, projection=ccrs.Robinson())
        self.ax = self.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        self.extent = [-180, 180, -75, 75]
        self.start_extent = True

        self.ax.set_extent(self.extent, crs=ccrs.PlateCarree())

        # self.ax.stock_img()
        # ax.add_feature(cfeature.LAND.with_scale('110m'))
        # ax.add_feature(cfeature.COASTLINE.with_scale('110m'))
        # ax.add_feature(cfeature.BORDERS, linestyle='--')
        # ax.add_feature(cfeature.LAKES, alpha=0.5)
        # ax.add_feature(cfeature.RIVERS)
        # self.ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle=':')

        self.ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
        self.ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle='--')
        self.ax.add_feature(cfeature.LAND.with_scale('50m'))
        self.ax.add_feature(cfeature.OCEAN.with_scale('50m'))

        self.ax.gridlines(draw_labels=True)

    def __del__(self):
        """
        Destructor: needed so that figure automatically close on rerun, etc...

        Returns
        -------
        None.

        """
        plt.close(self.fig)

    def addTrack(self, x, ac, z=None):
        """
        add the trace of a plane. Also adjusts the plot window extent

        Parameters
        ----------
        x : pd.DataFrame
            contains plane location info.
        ac : list of scalars
            The (multiple) aircraft to compute tracks for.
        z : pd.DataFrame, optional
            Ground Truth dataframe. The default is None.

        Returns
        -------
        None.

        """

        for c in ac:
            cur_id = x.loc[x.ac == c].index
            self.updateExtent(x.loc[x.ac == c, 'long'],
                              x.loc[x.ac == c, 'lat']
                              )
            self.ax.plot(x.loc[x.ac == c, 'long'],
                         x.loc[x.ac == c, 'lat'],
                         transform=ccrs.Geodetic()
                         )

            if (z is not None):
                self.ax.plot(z.loc[cur_id, 'long'],
                             z.loc[cur_id, 'lat'],
                             transform=ccrs.Geodetic()
                             )
                self.updateExtent(z.loc[cur_id, 'long'],
                                  z.loc[cur_id, 'lat']
                                  )

    def addPoint(self, x, id, z=None):
        """
        add a planes position as a point. Also adjusts the plot window extent.

        Parameters
        ----------
        x : pd.DataFrame
            contains plane location info.
        id : list of ints
            The (multiple) sample id's to plot the locatinos for
        z : pd.DataFrame, optional
            Ground Truth dataframe. The default is None.

        Returns
        -------
        None.

        """
        self.updateExtent(x.loc[id, 'long'],
                          x.loc[id, 'lat']
                          )

        if z is not None:
            self.updateExtent(z.loc[id, 'long'],
                              z.loc[id, 'lat']
                              )

        for c in id:
            self.ax.plot(x.loc[id, 'long'],
                         x.loc[id, 'lat'],
                         's',
                         transform=ccrs.Geodetic()
                         )

            if z is not None:
                self.ax.plot(z.loc[id, 'long'],
                             z.loc[id, 'lat'],
                             's',
                             transform=ccrs.Geodetic()
                             )

    def addPointByCoords(self, sp):
        """
        add a point just by coordinates. Also adjusts the plot window extent.

        Parameters
        ----------
        sp : array(n,2)
            contains lat and long as row vectors

        Returns
        -------
        None.

        """

        self.updateExtent(sp.T[1], sp.T[0])

        for lat, long in sp:
            self.ax.plot(long, lat, 'o', transform=ccrs.Geodetic())

    def addNode(self, nodes, ns):
        """
        Add a station node by its id's. Also adjusts the plot window extent.

        Parameters
        ----------
        nodes : pd.DataFrame
            The stations dataframe.
        ns : list of ints
            the id's of the stations to be plotted.

        Returns
        -------
        None.

        """

        for n in ns:
            self.updateExtent(nodes.at[n, 'long'],
                              nodes.at[n, 'lat']
                              )
            self.ax.plot(nodes.at[n, 'long'],
                         nodes.at[n, 'lat'],
                         '^',
                         transform=ccrs.Geodetic()
                         )

    def addNodeById(self, nodes, x, id):
        """
        Add all station nodes receiving the measurement "id". Also adjusts the
        plot window extent.

        Parameters
        ----------
        nodes : pd.DataFrame
            The stations dataframe.
        x : pd.DataFrame
            Measurements. contains plane location info.
        id : measurement id

        Returns
        -------
        None.

        """

        for c in id:
            tmp = np.array(x.at[c, 'n'])
            for n in tmp:
                self.updateExtent(nodes.at[n, 'long'],
                                  nodes.at[n, 'lat']
                                  )
                self.ax.plot(nodes.at[n, 'long'],
                             nodes.at[n, 'lat'],
                             '^',
                             transform=ccrs.Geodetic()
                             )

    def updateExtent(self, longs, lats):
        """
        Updates the lateral and vertical extend of the map plotted to added
        locations on the figure

        Parameters
        ----------
        longs : array(n, 1)
            Longs that were added.
        lats : array(n, 1)
            Lats that were added.

        Returns
        -------
        int
            DESCRIPTION.

        """

        # check for NaNs
        if np.isnan(longs).any() or np.isnan(lats).any():
            print("NaN")
            return 1

        # suppose the new extend based on the longs and lats passed to this
        # function:
        new_extent_raw = np.array([np.min(longs),
                                   np.max(longs),
                                   np.min(lats),
                                   np.max(lats)
                                   ])

        # see if this was the first time this function is called
        if not self.start_extent:
            # if not the first time: update only if it is actually bigger or
            # smaller than the already existing extent
            new_extent = [np.min([new_extent_raw[0] - 1, self.extent[0]]),
                          np.max([new_extent_raw[1] + 1, self.extent[1]]),
                          np.min([new_extent_raw[2] - 1, self.extent[2]]),
                          np.max([new_extent_raw[3] + 1, self.extent[3]])
                          ]
            self.extent = new_extent

        else:
            # if so, just use the extent based on the passed args plus margin
            self.extent = new_extent_raw + np.array([-1, 1, -1, 1])
            self.start_extent = False  # set first time flag to false

        # actually make the changes in the axis object
        self.ax.set_extent(self.extent, crs=ccrs.PlateCarree())
