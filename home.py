import geopandas as pd
import numpy as np
import pandas as pd
import streamlit as st #streamlit é a ferramenta que faz o site interativo para o modelo

from joblib import load

from notebooks.src.config import DADOS_LIMPOS, DADOS_GEO_MEDIAN, MODELO_FINAL

#@ simboliza uma função que atua sobre outra
# streamlit roda automatiamente tudo a cada alteração, o que pode demorar dependendo dos dados e funções
# Persistência de memória é usar um mecanismo que guarde info de modo não rerordar sempre
# .cache_x é função do streamlit que joga a função abaixo dela no cache (decorador)
@st.cache_data #guarda a função abaixo no cache de dados
def carregar_dados_limpos():
    return pd.read_parquet(DADOS_LIMPOS)

@st.cache_data
def carregar_dados_geo():
    return pd.read_parquet(DADOS_GEO_MEDIAN)

@st.cache_resource #guarda a função abaixo no cache de recursos
def carregar_modelo():
    return load(MODELO_FINAL)


df = carregar_dados_limpos() #recebe a info da funçaõ criada
gdf_geo = carregar_dados_geo() #recebe a info da funçaõ criada
modelo = carregar_modelo() #recebe a info da funçaõ criada


st.title("Previsão de preço de imóveis")

#criar caixas de input para cada uma das 13 features que o modelo precisa receber
#variaveis com mesmo nome que as features do modelo para o dict/df que iremos gerar tenha os nomes iguais
longitude = st.number_input("Longitude", value=-122.00)#cria caixa de input de número com padrão 122 e nome Longitude
latitude = st.number_input("Latitude", value=37.00)#cria caixa de input de número com padrão 122 e nome Latitude

housing_median_age = st.number_input("Idade do Imóvel", value=10)

total_rooms = st.number_input("Total de cômodos", value=800)
total_bedrooms = st.number_input("Total de quartos", value=100)
population = st.number_input("População", value=300)
households = st.number_input("Domicílios", value=100)

#cria slider com input de renda com mínimo 0.5, maximos 15, valor base 4.5 e passos/steps 0.5
median_income = st.slider("Renda média(múltiplos de US$10k)", 0.5, 15.0, 4.5, 0.5)

#cria caixa de seleção de proximidade que puxa opções únicas da feature do df
ocean_proximity = st.selectbox("Proximidade do oceano", df["ocean_proximity"].unique())

median_income_cat = st.number_input("Categoria de renda", value=4)

rooms_per_household = st.number_input("Cômodos por domicilio", value=7)
bedrooms_per_rooms = st.number_input("Razão de quartos por cômodo", value=0.2)
population_per_household = st.number_input("Pessoas por domicilio", value=2)

#variável que recebe os dados do input em forma de dict para dps virar df
#nomes das features igual ao nome das features do modelo
entrada_modelo = {
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean_proximity,
    "median_income_cat": median_income_cat,
    "rooms_per_household": rooms_per_household,
    "bedrooms_per_rooms": bedrooms_per_rooms,
    "population_per_household": population_per_household,
}
#gerando df a partir do dict gerado com os inputs
df_entrada_modelo = pd.DataFrame(entrada_modelo, index=[0]) #precisa de índice para o streamlit usar

#botão para gerar previsão. É true se clicado
botao_previsao = st.button("Prever preço")

#se o botão for true gera a previsão, ou seja, se clicar. Gera abaixo do botão em forma de array de arrays
if botao_previsao:
    preco = modelo.predict(df_entrada_modelo)#gera previsão
    #passa para o corpo do botão. Formatamos para extrair do array de arrays
    st.write(f"Preço previsto: US$ {preco[0][0]:.2f}")



