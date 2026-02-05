import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from xgboost import XGBClassifier
df = pd.read_csv("credit_risk_dataset.csv")


def limpar_texto(texto):
    if isinstance(texto,str):
        texto = texto.strip().lower()
        texto = texto.replace("í", "i").replace("ç", "c")
        texto = texto.replace("ã", "a")
        texto = texto.replace(",", ".")
    return texto

df["person_home_ownership"] = df["person_home_ownership"].apply(limpar_texto)
df["loan_intent"] = df["loan_intent"].apply(limpar_texto)
df["loan_grade"] = df["loan_grade"].apply(limpar_texto)
df["cb_person_default_on_file"] = df["cb_person_default_on_file"].apply(limpar_texto)

#Sempre limpar os null antes de chamar o enconderlabel
df = df.dropna()

codificador = LabelEncoder()
#para cada coluna no df pegas as tipo object menos a alvo e transforma em float64!

for coluna in df.columns:
    if df[coluna].dtype == "object" and coluna != "loan_status":

         
        df[coluna] = pd.Series(codificador.fit_transform(df[coluna]), index=df.index)


# ve se tem alguma coluna com tipo oject que não seja a alvo
for coluna in df.columns:
    if df[coluna].dtype == "object" and coluna != "loan_status":
        print(coluna)


#Define quais valores serão de x e y;
x = df.drop(["loan_status"], axis=1)
y = df["loan_status"]



x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2,random_state=1)


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

modelo_forest = RandomForestClassifier()
modelo_vizinho = KNeighborsClassifier()
modelo_xgb = XGBClassifier()

modelo_forest.fit(x_treino, y_treino)
modelo_vizinho.fit(x_treino, y_treino)
modelo_xgb.fit(x_treino, y_treino)


# Sempre renomear as variáveis de previsão
forest_previsao = modelo_forest.predict(x_teste)
vizinho_previsao = modelo_vizinho.predict(x_teste)
xgb_previsao = modelo_xgb.predict(x_teste)

print(f"Acurácia Random Forest: {acc(y_teste, forest_previsao):.2%}")
print(f"Acurácia KNN: {acc(y_teste, vizinho_previsao):.2%}")
print(f"Acurácia XGB: {acc(y_teste, xgb_previsao):.2%}")

coluna = list(x_teste.columns)
importancia = pd.DataFrame(index=coluna, data=modelo_xgb.feature_importances_)  
importancia = importancia * 100


importancia = pd.DataFrame({
    'coluna name': x.columns,
    'Importancia': modelo_xgb.feature_importances_
}).sort_values(by='Importancia', ascending=False)

print(importancia)