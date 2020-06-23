import numpy  as np
import pandas as pd

def m2np(x):
	
	a = np.array(list(map(int,list(map(float, str(x).replace("],[",",").replace("[","").replace("]","").split(","))) )))
	
	if a.size < 3:
		nparr = np.array([])
		return nparr,nparr,nparr
	elif a.size%3:
		exit("Input measurement list could not be processed properly.")
	else:
		nparr = a.reshape( int(a.size/3), 3 )
		return nparr[:,0],nparr[:,1],nparr[:,2]

def readMeasurements(filename):

	MR = pd.read_csv(filename, delimiter=',')
	MR = MR.rename(columns = {'timeAtServer':'t', 'aircraft':'ac', 'latitude':'lat', 'longitude':'long', 'baroAltitude':'baroAlt', 'geoAltitude':'geoAlt', 'numMeasurements':'M', 'measurements':'m'})
	# id (1 through 2,074,193) | server time | aircraft | lat | long | baroAlt | geoAlt | numMeasurements | measurements | reciever Nodes | nano seconds gps time | RSSI
	# id | t | ac | lat | long | baroAlt | geoAlt | M | m | n | ms | R
	
	MR["n"], MR["ns"], MR["R"] = zip(*MR.m.apply(m2np))
	
	return MR


def readNodes(filename):

	NR = pd.read_csv(filename, delimiter=',')
	NR = NR.rename(columns = {'serial':'n', 'latitude':'lat', 'longitude':'long', 'height':'geoAlt'})
	# id (1 through 2,074,193) | server time | aircraft | lat | long | baroAlt | geoAlt | numMeasurements | measurements | reciever Nodes | nano seconds gps time | RSSI
	# id | t | ac | lat | long | baroAlt | geoAlt | M | m | n | ms | R
		
	return NR


def readSolutions(filename):
	
	SR = pd.read_csv(filename, delimiter=',')
	SR = SR.rename(columns = {'latitude':'lat', 'longitude':'long', 'geoAltitude':'geoAlt'})
	
	return SR

