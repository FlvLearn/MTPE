# -*- coding: utf-8 -*-
import streamlit as st
import sidrapy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
import warnings

# Suprime alertas de descontinuação
warnings.filterwarnings('ignore')

# Configuração da página no Streamlit
st.set_page_config(
    page_title="Comparativo Desemprego vs INPC/IPCA",
    page_icon="📈",
    layout="wide"
)

st.title("COMPARATIVO DO EFEITO DO DESEMPREGO NO INPC E NO IPCA")
st.caption("**Grupo:** Flávio Melo, Adrielio Alvaro, Rodrigo Felix")

# --- Bloco 1 & 2 & 3: Coleta, Tratamento e Cache dos Dados ---
@st.cache_data
def carregar_e_tratar_dados():
    # Coletando os dados do SIDRA (Últimos 120 meses)
    raw_ipca = sidrapy.get_table(table_code="7060", territorial_level="1", ibge_territorial_code="all", variable="2265", period="last 120")
    raw_inpc = sidrapy.get_table(table_code="7063", territorial_level="1", ibge_territorial_code="all", variable="2292", period="last 120")
    raw_desemprego = sidrapy.get_table(table_code="6381", territorial_level="1", ibge_territorial_code="all", variable="4099", period="last 120")

    def limpar_dados(df, nome_valor):
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)

        coluna_tempo = 'Mês (Código)' if 'Mês (Código)' in df.columns else 'Trimestre Móvel (Código)'

        df = df[[coluna_tempo, 'Valor']].rename(columns={coluna_tempo: 'Data', 'Valor': nome_valor})
        df[nome_valor] = pd.to_numeric(df[nome_valor].str.replace(',', '.'), errors='coerce')
        df.dropna(subset=[nome_valor], inplace=True)
        df['Data'] = pd.to_datetime(df['Data'], format='%Y%m')
        return df

    df_ipca = limpar_dados(raw_ipca, 'IPCA_12M')
    df_inpc = limpar_dados(raw_inpc, 'INPC_12M')
    df_desemprego = limpar_dados(raw_desemprego, 'Desemprego')

    # Merge pela data
    df_macro = df_ipca.merge(df_inpc, on='Data').merge(df_desemprego, on='Data')
    df_macro = df_macro.sort_values('Data').reset_index(drop=True)

    # Criação de variáveis defasadas
    df_macro['Var_Desemprego_12M'] = df_macro['Desemprego'].diff(12)
    df_macro['Var_Desem_12M_Lag3'] = df_macro['Var_Desemprego_12M'].shift(3)

    df_clean = df_macro.dropna(subset=['Var_Desem_12M_Lag3', 'IPCA_12M', 'INPC_12M']).copy()
    return df_clean

# Executa o carregamento
with st.spinner("A carregar e a processar dados da API do SIDRA/IBGE..."):
    df_clean = carregar_e_tratar_dados()

# --- Bloco 4: Matriz de Correlação ---
st.divider()
st.subheader("CORRELAÇÃO: Choque Anual de Desemprego (Defasado 3m) vs Inflação 12m")

matriz_corr = df_clean[['Var_Desem_12M_Lag3', 'IPCA_12M', 'INPC_12M']].corr()
st.dataframe(matriz_corr.loc[['IPCA_12M', 'INPC_12M'], ['Var_Desem_12M_Lag3']], use_container_width=True)

st.markdown("""
* **Interpretação 1:** Valores negativos indicam que aumentos anuais de desemprego reduzem a inflação acumulada do período.
* **Interpretação 2:** O INPC apresenta maior correlação com o desemprego.
""")

# --- Bloco 5: Visualização dos Efeitos Defasados na Inflação ---
st.divider()
st.subheader("Visualização dos Efeitos Defasados na Inflação")

data_ini = df_clean['Data'].min().strftime('%b/%Y')
data_fim = df_clean['Data'].max().strftime('%b/%Y')
titulo_periodo = f"({data_ini} a {data_fim})"

# Gráficos de Regressão / Dispersão
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# IPCA
sns.regplot(data=df_clean, x='Var_Desem_12M_Lag3', y='IPCA_12M', color='blue', scatter_kws={'alpha': 0.5}, ax=ax1)
ax1.set_title(f'Impacto do Choque Anual de Desemprego no IPCA\n{titulo_periodo}', fontsize=11)
ax1.set_xlabel('Variação do Desemprego há 3 meses (Acumulado 12M em p.p.)')
ax1.set_ylabel('IPCA (Acumulado 12 meses, %)')

# INPC
sns.regplot(data=df_clean, x='Var_Desem_12M_Lag3', y='INPC_12M', color='orange', scatter_kws={'alpha': 0.5}, ax=ax2)
ax2.set_title(f'Impacto do Choque Anual de Desemprego no INPC\n{titulo_periodo}', fontsize=11)
ax2.set_xlabel('Variação do Desemprego há 3 meses (Acumulado 12M em p.p.)')
ax2.set_ylabel('INPC (Acumulado 12 meses, %)')

plt.tight_layout()
st.pyplot(fig1)  # Exibe no Streamlit
plt.close(fig1)  # Libera memória

# Gráfico de Linhas para Evolução Temporal
fig2, ax_line = plt.subplots(figsize=(14, 6))

ax_line.plot(df_clean['Data'], df_clean['IPCA_12M'], label='IPCA (Acum. 12M)', color='blue', linewidth=2)
ax_line.plot(df_clean['Data'], df_clean['INPC_12M'], label='INPC (Acum. 12M)', color='orange', linewidth=2)
ax_line.plot(df_clean['Data'], df_clean['Desemprego'], label='Taxa de Desemprego', color='green', linewidth=2, linestyle='--')

ax_line.set_title(f'Evolução Temporal do IPCA, INPC e Taxa de Desemprego\n{titulo_periodo}', fontsize=13, fontweight='bold')
ax_line.set_xlabel('Linha do Tempo (Data)', fontsize=11)
ax_line.set_ylabel('Percentual (%)', fontsize=11)
ax_line.grid(True, linestyle=':', alpha=0.6)
ax_line.legend(loc='upper right', fontsize=10, frameon=True)

plt.tight_layout()
st.pyplot(fig2)  # Exibe no Streamlit
plt.close(fig2)  # Libera memória

# --- Bloco 6: Regressão Linear e Conclusão ---
st.divider()
st.subheader("ANÁLISE DE SENSIBILIDADE (JANELA ANUALIZADA)")

# Cálculos
slope_ipca, _, r_ipca, p_ipca, _ = linregress(df_clean['Var_Desem_12M_Lag3'], df_clean['IPCA_12M'])
slope_inpc, _, r_inpc, p_inpc, _ = linregress(df_clean['Var_Desem_12M_Lag3'], df_clean['INPC_12M'])

col1, col2, col3 = st.columns(3)
col1.metric("Elasticidade IPCA", f"{slope_ipca:.4f}", f"R²: {r_ipca**2:.4f}")
col2.metric("Elasticidade INPC", f"{slope_inpc:.4f}", f"R²: {r_inpc**2:.4f}")
col3.metric("Diferença Absoluta", f"{abs(slope_ipca - slope_inpc):.4f}")

st.markdown("### Conclusão")

if abs(slope_inpc) > abs(slope_ipca):
    dif_perc = ((abs(slope_inpc) - abs(slope_ipca)) / abs(slope_ipca)) * 100
    st.success(f"O **INPC** apresentou maior sensibilidade às variações do desemprego do que o IPCA, sendo **{dif_perc:.1f}% superior**.")
elif abs(slope_ipca) > abs(slope_inpc):
    dif_perc = ((abs(slope_ipca) - abs(slope_inpc)) / abs(slope_inpc)) * 100
    st.success(f"O **IPCA** apresentou maior sensibilidade às variações do desemprego do que o INPC, sendo **{dif_perc:.1f}% superior**.")
else:
    st.info("Ambos apresentaram a mesma sensibilidade à variação da taxa de desemprego.")
