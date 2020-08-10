# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 13:15:46 2020

@author: Till
"""

from .helper import CART2SP, SP2CART

import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt


def ErrorHist(SEL, columns=4):

    HI = SEL.copy()
    HI.loc[HI['NormError'] == 0, "NormError"] = np.nan

    bins = np.concatenate([np.logspace(-1, 6, 15)])
    M_list = np.unique(HI['M'])

    rows = int(np.ceil(len(M_list) / columns))

    fig, axs = plt.subplots(rows, columns, sharey=False, sharex=True)
    fig.suptitle("Number of Stations per Measurement -- Histograms -- 826a8f",
                 fontsize=16
                 )

    hists = []

    for idx, m in enumerate(M_list):
        # for idx, m in enumerate([2, 3]):
        ax = axs[int(np.floor(idx / columns)), idx % columns]
        n, b, __ = ax.hist(HI.loc[HI['M'] == m, 'NormError'],
                           bins=bins,
                           density=False
                           )
        hists.append([m, n, b])
        # ax.set_xticks(ticks=bins[:-1]+0.5)
        ax.set_xscale('log')
        ax.grid()
        ax.set_title("SEL set 4 -- M = %d" % m)
        __ = ax.set_xlabel("2D Error")
        ax.set_ylabel("Datapoints")

    return hists, fig, axs


def ErrorCovariance(SEL, disc=None):

    def onpick3(event):
        ind = event.ind
        print('onpick3 scatter:', SEL[use].index[ind],
              np.take(x, ind), np.take(y, ind))

    use = SEL["NormError"] > 0
    yGT, x = zip(SEL.loc[use, ['fval_GT', 'NormError']].to_numpy().T)
    y, x = zip(SEL.loc[use, ['fval', 'NormError']].to_numpy().T)

    fig = plt.figure()
    if disc is not None:
        sc = plt.scatter(x, y, s=2, c=disc[use].astype(int), picker=True, label='Algorithm residual')
        # scGT = plt.scatter(x, yGT, s=2, c=SEL.loc[use, 'dim'], picker=True, label='Ground Truth residual')
        fig.colorbar(sc)
    else:
        sc = plt.scatter(x, y, s=2, c=SEL.loc[use, 'dim'], picker=True, label='Algorithm residual')
        # plt.scatter(x, yGT, s=2, c=SEL.loc[use, 'dim'], picker=True, label='Ground Truth residual')
        fig.colorbar(sc)
    
    ax = plt.gca()

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel("NormError [m]")
    ax.set_ylabel("Objective function residual")

    # ax.set_title("GT Residual vs Error")
    ax.set_title("GT Residual vs Error for M > 4 & no fval cutoff")

    ax.grid()

    ax.legend()

    fig.canvas.mpl_connect('pick_event', onpick3)


def StationErrorPlot(NR_corrector):

    fig = plt.figure()

    for i in range(len(NR_corrector.NR_corr)):
        sta = i + 1
        x = np.array(NR_corrector.NR_corr[sta - 1][0])
        y = np.array(NR_corrector.NR_corr[sta - 1][2])

        if len(y) == 0:
            continue

        plt.plot(x, y, label=str(sta))

    plt.legend()
    plt.grid()

    return fig


def HyperPlot(MR, SR, NR, idx, x_sph, inDict, SQfield=False, LevelExtent='large', labels=True):

    # initiate plane plot
    pp = PlanePlot()

    # plot ground truth
    if np.isnan(MR.at[idx, "lat"]):
        pp.addPoint(SR, [idx])
    else:
        pp.addPoint(MR, [idx])

    # plot solution
    __ = pp.addPointByCoords(np.array([x_sph[0:2]]))

    # plot stations
    pp.addNodeById(NR, MR, [idx])
    
    # get ground truth
    if SR.index.intersection([idx]).astype(bool).any():
        x_sph_GT = SR.loc[idx, ['lat', 'long', 'geoAlt']]
    else:
        x_sph_GT = MR.loc[idx, ['lat', 'long', 'geoAlt']]

    # calculate scalar fields over the current extend of the plot
    n_vals = 50
    
    # decide extent
    if LevelExtent == 'large':
        longl, longu, latl, latu = pp.ax.get_extent()
    elif LevelExtent == 'small':
        longl, longu, latl, latu = (min(x_sph[1], x_sph_GT[1]) - 0.25,
                                    max(x_sph[1], x_sph_GT[1]) + 0.25,
                                    min(x_sph[0], x_sph_GT[0]) - 0.25,
                                    max(x_sph[0], x_sph_GT[0]) + 0.25,
                                    )
    
    # compute grid
    long, lat = np.meshgrid(np.linspace(longl,
                                        longu, n_vals),
                            np.linspace(latl,
                                        latu, n_vals)
                            )
    h = MR.at[idx, 'baroAlt'] * np.ones_like(long)
    x = np.zeros_like(long)
    y = np.zeros_like(long)
    z = np.zeros_like(long)
    
    for i, __ in enumerate(long):
        x[i], y[i], z[i] = SP2CART(np.array([lat[i], long[i], h[i]]).T).T

    F = np.zeros([inDict['dim'], n_vals, n_vals])
    Fsq = np.zeros([n_vals, n_vals])
    for i in range(n_vals):
        for j in range(n_vals):
            xvec = np.array([x[i, j], y[i, j], z[i, j]])
            F[:, i, j] = inDict['fun'](xvec, -1)
            if SQfield:
                Fsq[i, j] = inDict['fun'](xvec, 0)

    # plot combined square objective
    if SQfield:
        cs = pp.ax.contour(long, lat, Fsq,
                           np.logspace(1,  # levels
                                       np.ceil(np.log10(np.max(Fsq))),
                                       50),
                           transform=ccrs.PlateCarree(),
                           )
        # pp.ax.clabel(cs, fontsize=10)

    # plot indivudial measurements
    Ns = np.array(MR.loc[idx, 'n'])
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    i = 0
    for i in range(inDict['dim']):
        # for i in [0, 1, 2, 3, 4, 5, 6, 7, 9]:
        cs = pp.ax.contour(long, lat, F[i], [0],
                           transform=ccrs.PlateCarree(),
                           colors=colors[i % len(colors)],
                           )
        Ns2 = Ns[inDict['mp'][i]]
        label = "%d-%d" % (Ns2[0], Ns2[1])
        if labels:
            pp.ax.clabel(cs, fontsize=10, fmt=label)

    # plot history
    xhist = np.array(inDict["xlist"])
    xhist_sph = CART2SP(xhist)
    pp.ax.plot(xhist_sph[:, 1], xhist_sph[:, 0],
               transform=ccrs.PlateCarree(),
               color='k',
               marker='.',
               )

    return pp


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

    def addTrack(self, x, ac, z=None, color='red'):
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

        def onpick3(event, indexlist):
            ind = event.ind
            print('Index:', indexlist[ind])

        cur_ids = []
        for i, c in enumerate(ac):
            cur_ids.append(x.loc[x.ac == c].index)
            if (z is not None):
                cur_ids[i] = z.index.intersection(cur_ids[i])

            self.updateExtent(x.loc[cur_ids[i], 'long'],
                              x.loc[cur_ids[i], 'lat']
                              )
            self.ax.plot(x.loc[cur_ids[i], 'long'],
                         x.loc[cur_ids[i], 'lat'],
                         transform=ccrs.Geodetic(),
                         color=color, marker='.', markersize=4
                         )
            self.ax.scatter(x.loc[cur_ids[i], 'long'][x.loc[cur_ids[i], 'MLAT']],
                            x.loc[cur_ids[i], 'lat'][x.loc[cur_ids[i], 'MLAT']],
                            transform=ccrs.Geodetic(),
                            color="black", marker='+',
                            picker=True
                            )
            self.fig.canvas.mpl_connect('pick_event',
                                        lambda y:
                onpick3(y, x.loc[cur_ids[i], 'long'][x.loc[cur_ids[i], 'MLAT']]
                                                .index
                        ))

            if (z is not None):
                self.ax.plot(z.loc[cur_ids[i], 'long'],
                             z.loc[cur_ids[i], 'lat'],
                             transform=ccrs.Geodetic(),
                             color="green", marker='.', markersize=4
                             )
                self.updateExtent(z.loc[cur_ids[i], 'long'],
                                  z.loc[cur_ids[i], 'lat']
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
                self.ax.annotate(
                    str(n),
                    xy=(nodes.at[n, 'long'] + 0.05, nodes.at[n, 'lat'] + 0.05)
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
