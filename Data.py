import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

def Get_Data(): 
    file = 'C:\\Users\\gaeta\\Documents\\Code\\Python\\Trading\\DataFile\\bitcoin_2019-1-1_2021-3-2.csv'
    df = pd.read_csv(file) 
    return df["Open"].to_numpy() 

def Get_Vector_Legendre(Data, order) : 
    z_fit = np.polynomial.legendre.Legendre.fit(np.linspace(0, len(Data), num=len(Data)), Data, order)
    coeff = z_fit.convert().coeff  
    return coeff

def Get_Data_Points(Data, window_size, step_size) : 
    Data_Matrix = None 
    for j in range(0, window_size//step_size, step_size) : 
        DataPoints = None 
        for i in range(j, len(Data), window_size) : 
            D = np.expand_dims( Data[i : i+window_size], 0 )  
            if DataPoints == None : 
                DataPoints = D 
            else : 
                np.concatenate([DataPoints, D], axis=0, out=DataPoints)  

    DP = np.expand_dims(DataPoints, 0)
    if Data_Matrix == None : 
        Data_Matrix = DP 
    else : 
        np.concatenate([Data_Matrix, DP ], axis=0, out=Data_Matrix)

    return Data_Matrix 

def Get_Node(Grid, v) : 
    k = 0 
    try : 
        k = np.where(np.dot(Grid, v) / np.power(np.norm(v), 2) == 1)[0]
    except IndexError : 
        k = -1 
    return k 

def Creation(Data_Matrix, order, grid_size) : 
    node_vector = list()
    Edge_Matrix = np.array([[0]])
    Grid_Matrix = None  
    for DP in Data_Matrix : 
        for d in DP : 
            v = Get_Vector_Legendre(d, order) // grid_size
            
            k = Get_Node(Grid, v) 
            if k == -1 : 
                if Grid_Matrix == None : 
                    Grid_Matrix = np.expand_dims(v, 0) 
                else : 
                    np.concatenate([Grid_Matrix, np.expand_dims(v, 0)], axis=0, out=Grid_Matrix) 

                node_vector.append(0) 
                Edge_Matrix = np.pad(Edge_Matrix, ((0, previous_k - Edge_Matrix.shape(0)), (0, k - Edge_Matrix.shape(1)) ), "constant", constant_values=(0)) 
            else : 
               node_vector[k] += 1
               Edge_Matrix[previous_k, k] += 1 
                
    return Grid_Matrix, Edge_Matrix, node_vector

def Prediction(prob_from_j_to, Grid_Matrix, node_list) : 
    res = 0
    for k in node_list : 
        v = prob_from_j_to[k] * Grid_Matrix[k, :]
        if res == 0 : 
            res = v
        else : 
            res += v 
    return res

def prob1(Edge_Matrix, from_node) : 
    prob = dict() 
    node_list = list(np.where(Edge_Matrix[from_node, :] != 0))
    deno = np.sum(Edge_Matrix[from_node, :], axis=-1) 
    for k in node_list  : 
        prob[k] = Edge_Matrix[from_node, k] / deno 

    return prob, node_list

def main() : 
    data = Get_Data() 
    Data_Matrix = Get_Data_Points(data, 10, 3) 
    train = Data_Matrix[:, :-2, :]
    test= Data_Matrix[0,-2:, :]
    G, E, n = Creation(train, 6, 1)
    #prediction 
    p, nodes = prob1(E, Get_Vector_Legendre(test[0, :], 6)// 1) 
    vprime  = Prediction(p, G, nodes) 
    pred = np.polynomial.legendre.legval(np.arange(10, 20, 10), vprime)
    plt.plot(pred, label="pred")
    plt.plot(test[1, :], lable="truth")
    plt.legend(loc="upper right")
    plt.show()

main() 