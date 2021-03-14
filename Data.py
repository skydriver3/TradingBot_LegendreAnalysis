import pandas as pd 
import numpy as np 

def Get_Data(): 
    file = 'C:\\Users\\gaeta\\Documents\\Code\\Python\\Trading\\DataFile\\bitcoin_2019-1-1_2021-3-2.csv'
    df = pd.read_csv(file) 
    return df["open"]

def Get_Vector_Legendre(Data, order) : 
    z_fit = np.polynomial.legendre.Legendre.fit(np.linspace(0, len(Data), num=len(Data)), Data, order)
    coeff = z_fit.convert().coeff  
    return coeff

def Get_Data_Points(Data, window_size, step_size) : 
    Data_Matrix = None 
    for j in range(0, window_size//step_size, grid_size) : 
       DataPoints = None 
       for i in range(j, len(Data), window_size) : 
           D = np.expand_dims( Data[i : i+window_size], 0 )  
           if DataPoints == None : 
               DataPoints = D 
            else : 
               np.concatenate([DataPoints, D], axis=0, out=DataPoints)  

    DP = np.expand_dims(DataPoints, 0)
    if Data_Matrix = None : 
        Data_Matrix = DP 
    else : 
        np.concatenate([Data_Matrix, DP ], axis=0, out=Data_Matrix)

    return Data_Matrix 

def Get_Node(Grid, v) : 
    k = np.where(np.dot(Grid, v) / np.power(np.norm(v), 2) == 1)[0]
    return k 

def Creation(Data_Matrix, order, grid_size) : 
    node_vector = list()
    Edge_Matrix = np.array([[0]])
    Grid_Matrix = None  
    for DP in Data_Matrix : 
        for d in DP : 
            v = Get_Vector_Legendre(d, order) % grid_size
            
            k = Get_Node(Grid, v) 