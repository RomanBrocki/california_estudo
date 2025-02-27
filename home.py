import geopandas as gpd  # Biblioteca para manipulação de dados espaciais
import numpy as np  # Biblioteca para operações numéricas
import pandas as pd  # Biblioteca para manipulação de dados tabulares
import pydeck as pdk  # Biblioteca de visualização de mapas interativos usada no Streamlit
import streamlit as st  # Biblioteca para criar interfaces web interativas

from joblib import load  # Para carregar modelos previamente treinados

from notebooks.src.config import DADOS_LIMPOS, DADOS_GEO_MEDIAN, MODELO_FINAL

# Decorador @st.cache_data permite armazenar os resultados da função em cache.
# Isso evita a necessidade de recarregar os dados sempre que a página for atualizada.
@st.cache_data
def carregar_dados_limpos():
    return pd.read_parquet(DADOS_LIMPOS)

@st.cache_data
def carregar_dados_geo():
    return pd.read_parquet(DADOS_GEO_MEDIAN)

# @st.cache_resource faz cache do modelo carregado, evitando que ele seja recarregado sempre.
@st.cache_resource
def carregar_modelo():
    return load(MODELO_FINAL)

# Carrega os dados e o modelo uma única vez e os mantém em cache.
df = carregar_dados_limpos()
gdf_geo = carregar_dados_geo()
modelo = carregar_modelo()

# Define o título da interface
st.title("Previsão de preço de imóveis")

# Criamos uma lista com os nomes dos condados, ordenados alfabeticamente.
# Essa lista será usada como opções para seleção do usuário.
condados = list(gdf_geo["name"].sort_values())

# Criamos um layout de duas colunas para separar os inputs e o mapa
coluna1, coluna2 = st.columns(2)

# with é designador de contexto
# Seção de entrada de dados pelo usuário
with coluna1:
    # Seletor para o usuário escolher um condado
    selecionar_condado = st.selectbox("Condado", condados)

    # Pegamos as coordenadas medianas do condado selecionado no geodataframe.
    # O `@` antes da variável indica que estamos referenciando uma variável Python dentro da string do `query()`.
    longitude = gdf_geo.query("name == @selecionar_condado")["longitude"].values
    latitude = gdf_geo.query("name == @selecionar_condado")["latitude"].values

    # Campo de entrada para idade do imóvel (slider numérico)
    housing_median_age = st.number_input("Idade do Imóvel", value=10, min_value=1, max_value=50)

    # Obtendo os valores médios de atributos relevantes do condado selecionado
    total_rooms = gdf_geo.query("name == @selecionar_condado")["total_rooms"].values
    total_bedrooms = gdf_geo.query("name == @selecionar_condado")["total_bedrooms"].values
    population = gdf_geo.query("name == @selecionar_condado")["population"].values
    households = gdf_geo.query("name == @selecionar_condado")["households"].values

    # Slider para a renda média (valor ajustado para milhares de dólares)
    median_income = st.slider("Renda média (milhares de US$)", 5.0, 100.0, 45.0, 5.0)

    # Pegamos a variável categórica `ocean_proximity`, definida pela moda dos registros do condado
    ocean_proximity = gdf_geo.query("name == @selecionar_condado")["ocean_proximity"].values

    # Criamos os bins usados para categorizar a renda
    bins_income = [0, 1.5, 3, 4.5, 6, np.inf]

    # np.digitize classifica o valor da renda em uma das categorias definidas pelos bins.
    # Como alteramos a escala da renda no input (de 10k para 1k), precisamos dividir por 10 antes da classificação.
    median_income_cat = np.digitize(median_income / 10, bins=bins_income)

    # Obtendo outras métricas relacionadas à densidade habitacional do condado selecionado
    rooms_per_household = gdf_geo.query("name == @selecionar_condado")["rooms_per_household"].values
    bedrooms_per_rooms = gdf_geo.query("name == @selecionar_condado")["bedrooms_per_rooms"].values
    population_per_household = gdf_geo.query("name == @selecionar_condado")["population_per_household"].values

    # Criamos um dicionário com as entradas do usuário para transformar em DataFrame
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

    # Transformamos o dicionário em DataFrame (precisa de um índice para ser aceito pelo modelo)
    df_entrada_modelo = pd.DataFrame(entrada_modelo, index=[0])

    # Criamos um botão que, ao ser pressionado, gera a previsão do preço do imóvel
    botao_previsao = st.button("Prever preço")

    # Se o botão for pressionado, realizamos a previsão com o modelo carregado
    if botao_previsao:
        preco = modelo.predict(df_entrada_modelo)
        st.write(f"Preço previsto: US$ {preco[0][0]:.2f}")

# Seção de exibição do mapa
with coluna2:
    # Criamos um estado inicial para o mapa
    view_state = pdk.ViewState(
        #Se passarmos as variaveis de lat e long direto dá erro pois é array e está em otimização de float.
        #para acertar selecionamos o item do array de 1(0) e convertemos para float não otimizado com float()
        latitude=float(latitude[0]),  # Posição inicial do mapa (latitude)
        longitude=float(longitude[0]),  # Posição inicial do mapa (longitude)
        zoom=5,  # Nível inicial de zoom
        min_zoom=4,  # Nível mínimo de zoom permitido
        max_zoom=8,  # Nível máximo de zoom permitido
    )

    # Criamos o mapa interativo com as configurações iniciais
    mapa = pdk.Deck(
        initial_view_state=view_state,  # Usa as coordenadas e zoom definidos acima
        map_style="light",  # Define o estilo visual do mapa
    )

    # Exibimos o mapa dentro do Streamlit
    st.pydeck_chart(mapa)


