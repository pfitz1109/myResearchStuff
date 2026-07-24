import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# names of files to import
sparseImportFile = 'sparseParaviewSpreadsheet.csv'
denseImportFile = 'denseParaviewSpreadsheet.csv'

# import the sparse and dense .csv spreadsheets
# these are generated from ParaView, which converts the .pvd files and their
# data into .csv files 
print('Importing spreadsheet...')
dfSparse = pd.read_csv(sparseImportFile)
print('sparseImportFile successfully imported.')
dfDense = pd.read_csv(denseImportFile)
print('denseImportFile successfully imported.')

# grab some basic info about the data structure
denseRow, denseColumn = dfSparse.shape
sparseRow, sparseColumn = dfDense.shape

# column names are identical between the two dataframes, so we can use either
columnNames = dfDense.columns

"""
    grab only the columns we need to check if essential points are being constructed properly
    necessary info: pointID, j, lambda, x, t, u_d, u_essential
    # |u_d| : main checker, need to make sure that this is less than the magnitude of epsilon
    # pointID, j, lambda, x, t : all related to location information. also different rules if lambda=0
    if we find a point that is incorrectly labeled, we need its location
    # u_essential : this is what MRWT is telling us, we need to compare it
"""
denseEssentialInfo = dfDense.loc[:, ["Point ID", "j", "lambda", "x", "t", "u_d", "u_essential"]]
sparseEssentialInfo = dfSparse.loc[:, ["Point ID", "j", "lambda", "x", "t", "u_d", "u_essential"]]

# going to convert these to numpy arrays to more easily loop through them
denseEssentialMatrix = denseEssentialInfo.values
sparseEssentialMatrix = sparseEssentialInfo.values

# user-defined threshold for the simulation 
threshold = 0.01

# loop through and check each sparse point to make sure it is essential or not
for r in range(0,sparseRow-2):
    # have to reassign to zero at the start of each loop
    essential_test = 0
    # need to check value of lambda first - occurs in third column (second position)
    lambda_test = sparseEssentialMatrix[r, 2]
    # if lambda = 0, it's an essential point 
    if lambda_test == 0 :
        essential_test = 1
    # else, we have to check the magnitude of u_d
    else :
        coefMagnitude = np.abs(sparseEssentialMatrix[r, 5])
        # if the magnitude is smaller than epsilon, its an essential point
        if coefMagnitude > threshold :
            essential_test = 1
    # now we check to see if our logic matches the logic in the .pvd
    # u_essential is either 1 or 0, occurs in seventh column
    pvdEssential = sparseEssentialMatrix[r,6] 
    sparseCount = 0
    if pvdEssential == essential_test :
        continue
    else :
        pointID = sparseEssentialMatrix[r,0]
        x_coordinate = sparseEssentialMatrix[r,3]
        t_coordinate = sparseEssentialMatrix[r,4]
        print(f'Discrepancy found at following location: Point ID: {pointID}, x-coordinate: {x_coordinate}, t-coordinate: {t_coordinate}')
        print(f'Magnitude of Wavelet Coefficient: {coefMagnitude}, Lambda = {lambda_test}')
        print(f'Python script determined Essential Point = {essential_test}, .pvd reads {pvdEssential} \n')
        sparseCountcount = sparseCount + 1

print(f'Total number of discrepancies: {sparseCount}')