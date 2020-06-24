# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:26:06 2020

@author: Till
"""

import              numpy  as np
from   numpy import linalg as la
from geopy.distance import great_circle as gc

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

def twoErrorCalc(x, z, RMSEnorm = 2):
	global R0, X, Y, Z
	
	sol_idx_bool = np.in1d(x.id, z.id)
	N = z.id.size
	
	lat_x  = np.array(x[sol_idx_bool].lat)
	long_x = np.array(x[sol_idx_bool].long)
	
	lat_z  = np.array(z.lat)
	long_z = np.array(z.long)
	h_z    = np.array(z.geoAlt)
	
	norm_vec = np.zeros(1, N)
	
	for i in range(N):
		norm_vec = gc((long_x[i], lat_x[i]), (long_z[i], lat_z[i])).meters * (R0+h_z)/R0
	
	e = (np.sum(norm_vec**RMSEnorm)/N)**(1/RMSEnorm)
	
	return e


def threeErrorCalc(x, z, RMSEnorm = 2, pnorm = 2):
	global R0, X, Y, Z
	
	sol_idx_bool = np.in1d(x.id, z.id)
	N = z.id.size
	
	lat_x  = np.array(x[sol_idx_bool].lat)
	long_x = np.array(x[sol_idx_bool].long)
	h_x    = np.array(x[sol_idx_bool].geoAlt)
	
	lat_z  = np.array(z.lat)
	long_z = np.array(z.long)
	h_z    = np.array(z.geoAlt)
	
	cart_x = [X(lat_x, long_x, h_x), Y(lat_x, long_x, h_x), Z(lat_x, long_x, h_x)]
	cart_z = [X(lat_z, long_z, h_z), Y(lat_z, long_z, h_z), Z(lat_z, long_z, h_z)]
	
	norm_vec = la.norm(np.array(cart_z) - np.array(cart_x), pnorm, 0)
	
	e = (np.sum(norm_vec**RMSEnorm)/N)**(1/RMSEnorm)
	
	return e


def writeSolutions(filename, z):
	z.columns = ['id','latitude','longitude','geoAltitude']
	z.to_csv(filename, index = False)
	
	return 0


class PlanePlot():
    def __init__(self):
        self.fig = plt.figure(figsize=(15,8))
        self.ax  = self.fig.add_subplot(1,1,1, projection=ccrs.Robinson())
        
        self.extent = [-180, 180, -75, 75];
        self.start_extent = True
        
        self.ax.set_extent(self.extent, crs=ccrs.PlateCarree())
        
        #ax.stock_img()
        #ax.add_feature(cfeature.LAND.with_scale('110m'))
        #ax.add_feature(cfeature.COASTLINE.with_scale('110m'))
        #ax.add_feature(cfeature.BORDERS, linestyle='--')
        #ax.add_feature(cfeature.LAKES, alpha=0.5)
        #ax.add_feature(cfeature.RIVERS)
        #self.ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle=':')
        
        self.ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
        self.ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle='--')
        self.ax.add_feature(cfeature.LAND.with_scale('50m'))
        self.ax.add_feature(cfeature.OCEAN.with_scale('50m'))
        
    def addTrack(self, x, ac, z = None):
        
        for c in ac:
            cur_id = x[x.ac == c].id
            self.ax.plot(x[x.ac == c].long, x[x.ac == c].lat, transform=ccrs.Geodetic())
            self.updateExtent(x[x.ac == c].long, x[x.ac == c].lat)
            
            if (z is not None):
                self.ax.plot(z[np.in1d(z.id, cur_id)].long, z[np.in1d(z.id, cur_id)].lat, transform=ccrs.Geodetic())
                self.updateExtent(z[np.in1d(z.id, cur_id)].long, z[np.in1d(z.id, cur_id)].lat)
                   
    def addPoint(self, x, id, z = None):
        
        self.updateExtent(x[np.in1d(x.id, id)].long, x[np.in1d(x.id, id)].lat)
        if z is not None:
            self.updateExtent(z[np.in1d(z.id, id)].long, z[np.in1d(z.id, id)].lat)
        
        for c in id:
            self.ax.plot(x[x.id == c].long, x[x.id == c].lat, 'o', transform=ccrs.Geodetic())
            
            if z is not None:
                self.ax.plot(z[z.id == c].long, z[z.id == c].lat, 'o', transform=ccrs.Geodetic())
                
    def addNode(self, nodes, ns):
        
        for n in ns:
            self.updateExtent(nodes[nodes.n == n].long, nodes[nodes.n == n].lat)
            self.ax.plot(nodes[nodes.n == n].long, nodes[nodes.n == n].lat, 'x', transform=ccrs.Geodetic())
        
    def addNodeById(self, nodes, x, id):
        
        for c in id:
            tmp = np.array(x[x.id == c].n)
            for n in tmp[0]:
                print(n)
                self.updateExtent(nodes[nodes.n == n].long, nodes[nodes.n == n].lat)
                self.ax.plot(nodes[nodes.n == n].long, nodes[nodes.n == n].lat, 'x', transform=ccrs.Geodetic())
        
    def updateExtent(self, longs, lats):
        if np.isnan(longs).any() or np.isnan(lats).any():
            print("NaN")
            return 1
        
        new_extent_raw = np.array([np.min(longs), np.max(longs), np.min(lats), np.max(lats)])
        
        #new_extent     = 
        if not self.start_extent:
            new_extent     =[np.min( [ new_extent_raw[0]-1, self.extent[0] ] ), \
                             np.max( [ new_extent_raw[1]+1, self.extent[1] ] ), \
                             np.min( [ new_extent_raw[2]-1, self.extent[2] ] ), \
                             np.max( [ new_extent_raw[3]+1, self.extent[3] ] )]
            self.extent    = new_extent
                
        else:
            self.extent    = new_extent_raw + np.array([-1, 1, -1, 1])
            self.start_extent = False
            
        print (self.extent)
            
            
        self.ax.set_extent(self.extent, crs=ccrs.PlateCarree())
        
