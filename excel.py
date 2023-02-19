import numpy as np
import pandas as pd


rente = 0.02
inleg = 1000

pd = pd.DataFrame({"Jaar" : np.arange(1, 31)})
print(pd)

pd["Periode"] = 31 - pd["Jaar"]
print(pd)

pd["Rente"] = rente
print(pd)

pd["Factor"] = pd["Rente"] + 1 
print(pd)

pd["Periode Factor"] = pow(pd["Factor"], pd["Periode"])
print(pd)

pd["Inleg"] = inleg
print(pd)

pd["Future Value"] = pd["Periode Factor"] * pd["Inleg"] 
print(pd)

som = pd["Future Value"].sum()
print(som)

fact = pd["Periode Factor"].sum()
print(fact)

print(som/fact)
