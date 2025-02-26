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

# Para simplificar a interface, ao invés de pedir lat e long pediremos o condado, 
# e puxaremos a mediana a partir do geodataframe
# Criamos uma lista com os nomes dos condados, ordenados alfabeticamente.
# Isso será usado para exibir opções no select box do Streamlit.
condados = list(gdf_geo["name"].sort_values())

# Criamos um seletor no Streamlit onde o usuário pode escolher um condado da lista.
selecionar_condado = st.selectbox("Condado", condados)

# Usamos `query()` para filtrar o DataFrame e obter a longitude mediana do condado selecionado.
# O `@` antes da variável indica que estamos referenciando uma variável Python dentro da string do `query()`.
longitude = gdf_geo.query("name == @selecionar_condado")["longitude"].values

# Fazemos o mesmo para a latitude mediana do condado selecionado.
latitude = gdf_geo.query("name == @selecionar_condado")["latitude"].values


#minimo e máximo de acordo com os dados
housing_median_age = st.number_input("Idade do Imóvel", value=10, min_value=1, max_value=50)

# dados usados são a mediana que tem por condado no geodataframe já que são não intuitivos para quem usa
total_rooms = gdf_geo.query("name == @selecionar_condado")["total_rooms"].values
total_bedrooms = gdf_geo.query("name == @selecionar_condado")["total_bedrooms"].values
population = gdf_geo.query("name == @selecionar_condado")["population"].values
households = gdf_geo.query("name == @selecionar_condado")["households"].values

# para ficar mais amigável pedimos valor em milhares, 
# que depois dividiremos por 10 para ficar na mesma unidade que o modelo
# cria slider com input de renda com mínimo 5, maximos 150, valor base 4.5 e passos/steps 5
# não premite misturar float com inteiro
median_income = st.slider("Renda média(milhares de US$)", 5.0, 150.0, 45.0, 5.0)

#pegamos do geodf a ocean proximity, que foi criada usando a moda para as casas do condado
ocean_proximity = gdf_geo.query("name == @selecionar_condado")["ocean_proximity"].values

#os bins que usamos no cut para criar as cat no df original
bins_income = [0, 1.5, 3, 4.5, 6, np.inf]
#o np.digitize pega os dados, os bins, e retorna em qual cat os dados se encontram seguindo a ordem dos bins
# /10 pois para ficar mais amigável mudamos o pedido de renda para k, mas nos dados está em 10k
median_income_cat = np.digitize(median_income /10, bins=bins_income)

#pegamos do geodf a ocean proximity, que foi criada usando a mediana para as casas do condado
rooms_per_household = gdf_geo.query("name == @selecionar_condado")["rooms_per_household"].values
bedrooms_per_rooms = gdf_geo.query("name == @selecionar_condado")["bedrooms_per_rooms"].values
population_per_household = gdf_geo.query("name == @selecionar_condado")["population_per_household"].values


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
    "median_income": median_income / 10,
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



