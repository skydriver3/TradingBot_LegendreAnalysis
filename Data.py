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

def Get_list(Data, ln, jmp, order) : 
    ix = 0 
    lst = [] 
    while ix+(2*ln) < len(Data) :  
        tp = (None, None)
        tp[0] = Get_Vector_Legendre(Data[ix:ix+ln], order) 
        tp[1] = Get_Vector_Legendre(Data[ix+ln:ix+(2*ln)]) - tp[0]
        lst.append(tp) 
    
        ix += jmp 

    return lst 

def dist(v1, v2) : 
    return np.linalg.norm(v1 - v2) 

def Search_Nearest(v_list, vec, ln) : 
    v_list.sort(key=lambda x : dist(vec, x[0]))
    return v_list[:ln]

def Weighted_Avrg(K_nearest, vec, beta ) : 
   # Get jump_vec for each nearest point
   # Get 1/distance for each nearest point 
   # Dot product the 2 list  
   # Return the product 
    dist_v = []
    for v in K_nearest : 
        dist_v.append(beta / dist(vec, v[0]))
    
    dist_v = np.array(dist_v) 
    dist_v = dist_v / np.linalg.norm(dist_v)

    for v in K_nearest : 
        a = a + (v[1] / dist_v[i])

    return a 


def main() : 
    Data = Get_Data()

    ls = Get_list(Data, 10, 3, 7) 

    test = ls[-100:]
    
    hist = ls[:-100]

    pred = []
    for test_v in test : 
        nearest = Search_Nearest(hist, test_v[0], 100) 

        pred.append(Weighted_Avrg(nearest, test_v[0], 1))


    for ix, a in enumerate(pred) : 
         

    