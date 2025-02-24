import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import EngFormatter
from sklearn.metrics import PredictionErrorDisplay

from .models import RANDOM_STATE  # Importa a semente de aleatoriedade definida no módulo models

# Configuração global do Seaborn
sns.set_theme(palette="bright")

# Definição de constantes para a paleta de cores e transparência de pontos nos gráficos
PALETTE = "coolwarm"
SCATTER_ALPHA = 0.2

# ---------------------------------------------------------
# Plotagem dos coeficientes de um modelo de regressão
# ---------------------------------------------------------

def plot_coeficientes(df_coefs, titulo="Coeficientes"):
    """
    Gera um gráfico de barras horizontais para visualizar os coeficientes de um modelo.
    
    Parâmetros:
    - df_coefs: DataFrame contendo os coeficientes do modelo.
    - titulo: Título do gráfico.
    
    Retorna:
    - Exibe o gráfico de coeficientes.
    """
    
    df_coefs.plot.barh()  # Plota os coeficientes como um gráfico de barras horizontal
    plt.title(titulo)  # Define o título do gráfico
    plt.axvline(x=0, color=".5")  # Adiciona uma linha vertical no zero para referência
    plt.xlabel("Coeficientes")  # Define o rótulo do eixo X
    plt.gca().get_legend().remove()  # Remove a legenda do gráfico, se houver
    plt.show()  # Exibe o gráfico


# ---------------------------------------------------------
# Análise de resíduos do modelo
# ---------------------------------------------------------

def plot_residuos(y_true, y_pred):
    """
    Gera gráficos para análise dos resíduos do modelo.

    Parâmetros:
    - y_true: Valores reais da variável alvo.
    - y_pred: Valores preditos pelo modelo.

    Retorna:
    - Exibe histogramas e gráficos de resíduos.
    """

    residuos = y_true - y_pred  # Calcula os resíduos (erro entre valores reais e preditos)

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))  # Cria uma figura com três subgráficos

    # Histograma dos resíduos
    sns.histplot(residuos, kde=True, ax=axs[0])

    # Gráfico de resíduos vs valores preditos
    PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind="residual_vs_predicted", ax=axs[1]
    )

    # Gráfico de valores reais vs preditos
    PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind="actual_vs_predicted", ax=axs[2]
    )

    plt.tight_layout()  # Ajusta o espaçamento entre os gráficos
    plt.show()  # Exibe os gráficos


# ---------------------------------------------------------
# Análise de resíduos a partir de um modelo (estimador)
# ---------------------------------------------------------

def plot_residuos_estimador(estimator, X, y, eng_formatter=False, fracao_amostra=0.25):
    """
    Gera gráficos para análise dos resíduos do modelo a partir de um estimador.

    Parâmetros:
    - estimator: Modelo de regressão treinado.
    - X: Features do conjunto de dados.
    - y: Valores reais da variável alvo.
    - eng_formatter: Se True, formata os eixos com unidades de engenharia.
    - fracao_amostra: Proporção de amostras usada para os gráficos (padrão 25%).

    Retorna:
    - Exibe histogramas e gráficos de resíduos.
    """

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))  # Cria uma figura com três subgráficos

    # Gráfico de resíduos vs valores preditos
    error_display_01 = PredictionErrorDisplay.from_estimator(
        estimator,
        X,
        y,
        kind="residual_vs_predicted",
        ax=axs[1],
        random_state=RANDOM_STATE,
        scatter_kwargs={"alpha": SCATTER_ALPHA},  # Define a transparência dos pontos
        subsample=fracao_amostra,  # Usa uma fração dos dados para o gráfico
    )

    # Gráfico de valores reais vs preditos
    error_display_02 = PredictionErrorDisplay.from_estimator(
        estimator,
        X,
        y,
        kind="actual_vs_predicted",
        ax=axs[2],
        random_state=RANDOM_STATE,
        scatter_kwargs={"alpha": SCATTER_ALPHA},
        subsample=fracao_amostra,
    )

    # Calcula os resíduos a partir do gráfico gerado
    residuos = error_display_01.y_true - error_display_01.y_pred

    # Plota o histograma dos resíduos
    sns.histplot(residuos, kde=True, ax=axs[0])

    # Se ativado, formata os eixos usando o EngFormatter para melhor legibilidade
    if eng_formatter:
        for ax in axs:
            ax.yaxis.set_major_formatter(EngFormatter())
            ax.xaxis.set_major_formatter(EngFormatter())

    plt.tight_layout()  # Ajusta o espaçamento entre os gráficos
    plt.show()  # Exibe os gráficos


# ---------------------------------------------------------
# Comparação de métricas entre diferentes modelos
# ---------------------------------------------------------

def plot_comparar_metricas_modelos(df_resultados):
    """
    Gera gráficos comparativos das métricas de diferentes modelos de regressão.

    Parâmetros:
    - df_resultados: DataFrame contendo os resultados dos modelos.

    Retorna:
    - Exibe gráficos comparando tempo de execução e métricas como R², MAE e RMSE.
    """

    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True)  # Cria uma grade de 2x2 para os gráficos

    # Lista das métricas a serem comparadas
    comparar_metricas = [
        "time_seconds",  # Tempo de execução do modelo
        "test_r2",  # Coeficiente de determinação R²
        "test_neg_mean_absolute_error",  # Erro absoluto médio (MAE negativo)
        "test_neg_root_mean_squared_error",  # Raiz do erro quadrático médio (RMSE negativo)
    ]

    # Rótulos para os gráficos
    nomes_metricas = [
        "Tempo (s)",
        "R²",
        "MAE",
        "RMSE",
    ]

    # Itera sobre cada métrica e plota um gráfico de boxplot para comparação entre modelos
    for ax, metrica, nome in zip(axs.flatten(), comparar_metricas, nomes_metricas):
        sns.boxplot(
            x="model",  # Modelos no eixo X
            y=metrica,  # Métrica no eixo Y
            data=df_resultados,
            ax=ax,
            showmeans=True,  # Exibe a média como um marcador no gráfico
        )
        ax.set_title(nome)  # Define o título do gráfico
        ax.set_ylabel(nome)  # Define o rótulo do eixo Y
        ax.tick_params(axis="x", rotation=90)  # Rotaciona os nomes dos modelos para melhor visualização

    plt.tight_layout()  # Ajusta os espaçamentos
    plt.show()  # Exibe os gráficos

