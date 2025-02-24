import pandas as pd

def dataframe_coeficientes(coefs, colunas):
    """
    Cria um DataFrame organizado com os coeficientes de um modelo de regressão.

    Parâmetros:
    - coefs: Lista ou array contendo os coeficientes do modelo.
    - colunas: Lista com os nomes das variáveis correspondentes aos coeficientes.

    Retorna:
    - Um DataFrame com duas colunas:
        - "coeficiente": Valores dos coeficientes
        - Índice do DataFrame: Nomes das variáveis
      O DataFrame é ordenado em ordem crescente com base no valor do coeficiente.
    """

    return (
        pd.DataFrame(
            data=coefs,  # Define os coeficientes como dados do DataFrame
            index=colunas,  # Define os nomes das variáveis como índice
            columns=["coeficiente"]  # Nome da coluna onde os coeficientes serão armazenados
        )
        .sort_values(by="coeficiente")  # Ordena o DataFrame do menor para o maior coeficiente
    )

