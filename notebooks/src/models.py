import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42  # Define uma semente para reprodutibilidade dos resultados

# ---------------------------------------------------------
# Construção do pipeline para modelos de regressão
# ---------------------------------------------------------

def construir_pipeline_modelo_regressao(
    regressor, preprocessor=None, target_transformer=None
):
    """
    Constrói um pipeline de regressão opcionalmente incluindo um pré-processador e um transformador de alvo.
    
    Parâmetros:
    - regressor: modelo de regressão do sklearn.
    - preprocessor: transformações a serem aplicadas nos dados antes do modelo (ex: escalonamento, encoding).
    - target_transformer: transformador para aplicar ao alvo (ex: log-transform).
    
    Retorna:
    - Um pipeline de regressão com as transformações especificadas.
    """
    
    # Se houver um pré-processador, ele é adicionado antes do regressor no pipeline
    if preprocessor is not None:
        pipeline = Pipeline([("preprocessor", preprocessor), ("reg", regressor)])
    else:
        pipeline = Pipeline([("reg", regressor)])  # Apenas o regressor, sem pré-processamento

    # Se houver um transformador de alvo, aplica ao pipeline de regressão
    if target_transformer is not None:
        model = TransformedTargetRegressor(
            regressor=pipeline, transformer=target_transformer
        )
    else:
        model = pipeline  # Se não houver transformador, usa apenas o pipeline

    return model


# ---------------------------------------------------------
# Treinamento e validação do modelo com Cross-Validation
# ---------------------------------------------------------

def treinar_e_validar_modelo_regressao(
    X,
    y,
    regressor,
    preprocessor=None,
    target_transformer=None,
    n_splits=5,
    random_state=RANDOM_STATE,
):
    """
    Treina e valida um modelo de regressão usando K-Fold Cross-Validation.
    
    Parâmetros:
    - X: Features do conjunto de dados.
    - y: Variável alvo.
    - regressor: modelo de regressão.
    - preprocessor: opcional, transformações a serem aplicadas nas features.
    - target_transformer: opcional, transformador aplicado ao alvo (ex: log).
    - n_splits: número de divisões para Cross-Validation.
    - random_state: semente para reprodutibilidade.
    
    Retorna:
    - Dicionário com métricas de desempenho (R², MAE negativo, RMSE negativo).
    """
    
    # Cria o pipeline do modelo com pré-processamento e transformações opcionais
    model = construir_pipeline_modelo_regressao(
        regressor, preprocessor, target_transformer
    )

    # Configura o K-Fold Cross-Validation com embaralhamento dos dados
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Executa a validação cruzada, calculando métricas para cada divisão dos dados
    scores = cross_validate(
        model,
        X,
        y,
        cv=kf,
        scoring=[
            "r2",  # Coeficiente de determinação (quanto maior, melhor o ajuste)
            "neg_mean_absolute_error",  # Erro absoluto médio (MAE) negativo
            "neg_root_mean_squared_error",  # Raiz do erro quadrático médio (RMSE) negativo
        ],
    )

    return scores


# ---------------------------------------------------------
# Busca de hiperparâmetros via GridSearchCV
# ---------------------------------------------------------

def grid_search_cv_regressor(
    regressor,
    param_grid,
    preprocessor=None,
    target_transformer=None,
    n_splits=5,
    random_state=RANDOM_STATE,
    return_train_score=False,
):
    """
    Realiza uma busca de hiperparâmetros usando GridSearchCV.
    
    Parâmetros:
    - regressor: modelo de regressão a ser otimizado.
    - param_grid: dicionário de parâmetros a serem testados no GridSearchCV.
    - preprocessor: opcional, transformações a serem aplicadas nas features.
    - target_transformer: opcional, transformador aplicado ao alvo (ex: log).
    - n_splits: número de divisões para Cross-Validation.
    - random_state: semente para reprodutibilidade.
    - return_train_score: se True, retorna métricas do conjunto de treino.
    
    Retorna:
    - Objeto GridSearchCV configurado para encontrar os melhores hiperparâmetros.
    """
    
    # Cria o pipeline do modelo com pré-processamento e transformações opcionais
    model = construir_pipeline_modelo_regressao(
        regressor, preprocessor, target_transformer
    )

    # Configura o K-Fold Cross-Validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Executa a busca em grade dos melhores hiperparâmetros
    grid_search = GridSearchCV(
        model,
        param_grid=param_grid,  # Conjunto de hiperparâmetros a testar
        cv=kf,  # Validação cruzada
        scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],  # Métricas avaliadas
        refit="neg_root_mean_squared_error",  # Define a métrica principal para ajuste do melhor modelo
        n_jobs=-1,  # Usa todos os núcleos disponíveis para acelerar o processamento
        return_train_score=return_train_score,  # Retorna também as métricas no conjunto de treino
        verbose=1,  # Exibe informações do progresso da busca
    )

    return grid_search


# ---------------------------------------------------------
# Organização dos resultados do GridSearchCV
# ---------------------------------------------------------

def organiza_resultados(resultados):
    """
    Organiza os resultados da validação cruzada em um DataFrame mais estruturado.
    
    Parâmetros:
    - resultados: dicionário contendo os resultados da validação cruzada.
    
    Retorna:
    - DataFrame com as métricas organizadas e tempo total de execução.
    """

    # Adiciona uma nova coluna "time_seconds" somando o tempo de ajuste (fit) e avaliação (score)
    for chave, valor in resultados.items():
        resultados[chave]["time_seconds"] = (
            resultados[chave]["fit_time"] + resultados[chave]["score_time"]
        )

    # Converte o dicionário de resultados em um DataFrame
    df_resultados = (
        pd.DataFrame(resultados).T.reset_index().rename(columns={"index": "model"})
    )

    # Expande os resultados para que cada métrica tenha uma linha própria
    df_resultados_expandido = df_resultados.explode(
        df_resultados.columns[1:].to_list()
    ).reset_index(drop=True)

    # Converte valores para numéricos quando possível
    try:
        df_resultados_expandido = df_resultados_expandido.apply(pd.to_numeric)
    except ValueError:
        pass  # Ignora erro caso alguma coluna não possa ser convertida

    return df_resultados_expandido
