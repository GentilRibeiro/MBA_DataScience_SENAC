# 1. IMPORTAR BIBLIOTECAS

import streamlit as st
from streamlit_folium import folium_static
import folium
from folium.plugins import HeatMap
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import networkx as nx
import matplotlib.pyplot as plt

# 2. CONFIGURAÇÃO DA PÁGINA

st.set_page_config(page_title="Dashboard", layout="wide")

# 3. TÍTULO DASHBOARD

with st.container():
    st.write("<h4 style='color:green; font-size:20px;'> Faculdade SENAC - PE * MBA Ciência de Dados e IA * Linguagem Python (Prof. Geraldo Gomes) * Agosto 2024</h4>", 
    unsafe_allow_html=True)
    st.title("Análise de Sinistros de Trânsito - Cidade do Recife")
    st.write("Série histórica: 2016 a 2024")
    st.write("Fonte: Autarquia de Trânsito e Transporte Urbano do Recife - CTTU")
    st.write("Quer acessar a fonte de dados?  [Clique aqui](https://dados.recife.pe.gov.br/pt_PT/dataset/acidentes-de-transito-com-e-sem-vitimas)")
      
# 4. CARREGAMENTO E TRATAMENTO

with st.container():
    st.write("---")
    dados = pd.read_csv("consolidado.csv", delimiter=",")
    dados['data'] = pd.to_datetime(dados['data'])

# 5. CRIANDO UM MENU LATERAL
    
    # Adicionar imagem acima do nome "Filtros"
    st.sidebar.image("https://i.postimg.cc/Y9r35LCg/Logo-1.png", width=250)
    
    # Barra - marcador
    st.sidebar.header("Filtros")  

    # Filtro tipo de acidentes
    tipos_unicos = dados['tipo'].unique()
    tipo_acid = st.sidebar.multiselect("Escolha o tipo do acidente", tipos_unicos)

    # Obter a data mínima e máxima após a conversão correta para datetime
    dt_inicio = dados['data'].min().to_pydatetime()
    dt_fim = dados['data'].max().to_pydatetime()
    # Criar o slider de intervalo de datas com valores do tipo datetime
    intervalo_datas = st.sidebar.slider(
        "Selecione o intervalo de datas",
        min_value=dt_inicio,
        max_value=dt_fim,
        value=(dt_inicio, dt_fim)
    )
    # Aplicar o filtro de datas nos dados
    dados = dados[(dados['data'] >= intervalo_datas[0]) & (dados['data'] <= intervalo_datas[1])]

    # Aplica filtro aos dados
    if tipo_acid:
        dados = dados[dados['tipo'].isin(tipo_acid)]

    st.sidebar.write("---")
    
    # Barra - marcador
    st.sidebar.header("Equipe")  

    # Adicionar imagem acima do nome "Filtros"
    st.sidebar.image("https://i.postimg.cc/7ZPSYrk8/Equipe.png", width=200)

# 6. PRÉ-PROCESSAMENTO

    # Remove colunas 100% sem dados
    dados = dados.dropna(axis=1, how='all')
    # Função para formatar números com separador de milhar
    def format_number(number):
        try:
            return "{:,.0f}".format(float(number)).replace(',', '.')
        except (ValueError, TypeError):
            return number  # Retorna o valor original se ocorrer um erro
    # Convertendo texto para números inteiros
    dados['vitimas'] = pd.to_numeric(dados['vitimas'], errors='coerce')
    dados['vitimasfatais'] = pd.to_numeric(dados['vitimasfatais'], errors='coerce')
    # Calcular totais
    total_vitimas = dados['vitimas'].sum()
    total_vitimas_fatais = dados['vitimasfatais'].sum()
    total_ocorrencias = dados['data'].count()
    # Criar colunas
    col1, col2, col3 = st.columns(3)
    # Formatação dos números
    col1.markdown(f"<h2 style='font-size: 24px; color: #0958D9;'>Total de Ocorrências</h2><p style='font-size: 50px;'>{format_number(total_ocorrencias)}</p>", unsafe_allow_html=True)
    col2.markdown(f"<h2 style='font-size: 24px; color: #0958D9;'>Total de Vítimas</h2><p style='font-size: 50px;'>{format_number(total_vitimas)}</p>", unsafe_allow_html=True)
    col3.markdown(f"<h2 style='font-size: 24px; color: #0958D9;'>Total de Vítimas Fatais</h2><p style='font-size: 50px;'>{format_number(total_vitimas_fatais)}</p>", unsafe_allow_html=True)

# 7. AGREGAÇÕES E DATA VISUALIZATION

    st.write("---")
    
    # Total de ocorrências por ano (Gráfico de Barras Vertical)
    st.write("Total de Ocorrências por Ano")
    dados['ano'] = dados['data'].dt.year
    ocorrencias_por_ano = dados.groupby('ano')['data'].count().reset_index()
    ocorrencias_por_ano.columns = ['Ano', 'Total de Ocorrências']
    fig_barras = px.bar(
        ocorrencias_por_ano,
        x='Ano',
        y='Total de Ocorrências',
        labels={'Total de Ocorrências': 'Total de Ocorrências'},
        text_auto=True,
        color_discrete_sequence=['#0958D9']
    )  # Criar o gráfico de barras
    fig_barras.update_layout(
       yaxis=dict(
           tickformat=".0f"
       )
    ) # Atualizar layout para formatar os números com ponto de milhar e sufixo "k"
    st.plotly_chart(fig_barras, use_container_width=True) # Exibir o gráfico no Streamlit
    st.write("Insights:")
    st.write("1. Houve uma consideralvel redução nos niveis de ocorrências considerando os anos de 2019 versus 2020, possivelmente objeto de novas políticas públicas de trânsito voltadas a redução da velocidade, melhor sinalização de trânsito, etc.")
    
    st.write("---")
        
    # Total de ocorrência por mês (Gráfico de Barras Vertical)
    st.write("Total de Ocorrências por Mês")
    dados['mes'] = dados['data'].dt.month  # Extrair o mês a partir da data
    ocorrencias_por_mes = dados.groupby('mes')['data'].count().reset_index()
    fig_mes = px.bar(ocorrencias_por_mes, x='mes', y='data', labels={'data': 'Total de Ocorrências', 'mes': 'Mês'}, 
                     text_auto=True, color_discrete_sequence=['#0958D9'])
    fig_mes.update_layout(xaxis=dict(
        tickmode='array',
        tickvals=list(range(1, 13)),
        ticktext=['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
                  'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    ))  # Ajustando o rótulo do eixo x para exibir os nomes dos meses
    st.plotly_chart(fig_mes)  # Exibir o gráfico no Streamlit
    st.write("Insights:")
    st.write("1. Conforme o gráfico acima expõe, exista uma tendência de aumeto do número de ocorrências entre o final e o incício do ano, considerando toda a série histórica;")
    st.write("2. Nota-se ainda um padrão de redução nos níveis de ocorrência entre os meses de junho e julho.")
    
    st.write("---")
   
    # Total de ocorrência por bairro (Top 10 - Gráfico de Barras Horizontal)
    st.write("Top 10 Bairros com Mais Ocorrências")
    ocorrencias_por_bairro = dados.groupby('uf_cidade_bairro')['data'].count().reset_index()  # Agrupar dados por bairro e calcular o total de ocorrências
    top_10_bairros = ocorrencias_por_bairro.sort_values(by='data', ascending=False).head(10)  # Ordenar por número de ocorrências e selecionar o top 10
    fig_top_10 = px.bar(top_10_bairros,  # Criar gráfico de barras horizontal
                        x='data', 
                        y='uf_cidade_bairro', 
                        orientation='h',  # Define o gráfico como horizontal
                        labels={'data': 'Total de Ocorrências', 'uf_cidade_bairro': 'Bairro'},
                        text='data',    # Adiciona os rótulos dos valores
                        color_discrete_sequence=['#0958D9'])  # Define a cor do gráfico
    fig_top_10.update_layout(yaxis={'categoryorder': 'total ascending'}, 
                             xaxis_title="Total de Ocorrências",
                             yaxis_title="Bairro")  # Ajustar layout para melhorar a visualização
    st.plotly_chart(fig_top_10)  # Exibir o gráfico no Streamlit
    st.write("Insights:")
    st.write("1. Imbiribeira e Boa Viagem lideram com os maiores números de acidentes.")
   
    st.write("---")
    
    # Comparativo Total de Vitimas vs Total de Vitimas Fatais (Grafico de Linhas multiplas)
    with st.container():
        st.write("Comparação entre Total de Vítimas e Vítimas Fatais por Mês")
        anos_disponiveis = sorted(dados['data'].dt.year.unique())  # Extrair e ordenar os anos disponíveis para o slicer
        anos_selecionados = st.multiselect('Selecione os Anos', anos_disponiveis, default=anos_disponiveis, key='anos_selecionados_1')  # Adicionar o slicer para selecionar múltiplos anos
        dados_filtrados = dados[dados['data'].dt.year.isin(anos_selecionados)]  # Filtrar os dados com base nos anos selecionados
        vitimas_por_mes = dados_filtrados.groupby(dados_filtrados['data'].dt.strftime('%b'))[['vitimas', 'vitimasfatais']].sum().reset_index()  # Calcular o total de vítimas e vítimas fatais por mês
        vitimas_por_mes = vitimas_por_mes.sort_values(by='data')  # Ordenar os meses
        fig_vitimas = go.Figure()
        fig_vitimas.add_trace(go.Scatter(x=vitimas_por_mes['data'], y=vitimas_por_mes['vitimas'],
                                        mode='lines+markers+text', name='Total de Vítimas',
                                        text=vitimas_por_mes['vitimas'], textposition='top center',
                                        line=dict(color='#0958D9')))
        fig_vitimas.add_trace(go.Scatter(x=vitimas_por_mes['data'], y=vitimas_por_mes['vitimasfatais'],
                                        mode='lines+markers+text', name='Total de Vítimas Fatais',
                                        text=vitimas_por_mes['vitimasfatais'], textposition='top center',
                                        line=dict(color='green')))
        fig_vitimas.update_layout(xaxis_title="Mês", yaxis_title="Total", 
                                xaxis=dict(
                                    tickmode='array',
                                    tickvals=np.arange(1, 13),
                                    ticktext=['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun','Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']))
        st.plotly_chart(fig_vitimas, use_container_width=True) # Exibir o gráfico em widescreen
    st.write("Insights:")
    st.write("1. A base analisada segrega vitimas de vitimas fatais, não havendo assim a possíbilidade de duplicação dos dados a serem expostos;")
    st.write("2. Apesar da pequena relevância do números de vitimas fatais frente ao total de vitimas não fatais, o comparativo visa trazer a informação no contexto quê, o estado deve prover políticas que visam evitar ambos os parâmetros, cada um com seu nível de criticidade;")
    st.write("3. O mês de Setembro lidera com o número total de vítimas em todos os anos;")
    st.write("4. O mês de Fevereiro é o maior com vítimas fatais em todos os anos.")

    st.write("---")

    # Mapa de calor mostrando o total de ocorrências por bairro
    st.write("Mapa de Calor - Total de Ocorrências por Bairro")
    dados = dados.dropna(subset=['Latitude', 'Longitude']) # Remover linhas onde a latitude ou longitude estão ausentes
    mapa = folium.Map(location=[-8.0476, -34.8770], zoom_start=12)
    heat_data = [[row['Latitude'], row['Longitude']] for index, row in dados.iterrows()]
    HeatMap(heat_data).add_to(mapa) # Criar o mapa de calor
    folium_static(mapa) # Renderizar o mapa no Streamlit   
    st.write("Insights:")
    st.write("1. Fica evidente que a Zona Sul e o Centro da cidade do Recife concentram o maior número de ocorrências, cabendo assim uma avaliação posterior de fatores, cuja a finalidade seria prover ações de segurança específicas para essas áreas.")
   
    st.write("---")


# 8. MODELO DE MACHINE LEARNING PARA CLASSIFICAÇÃO DE RISCO

    # Interface do Streamlit
    st.title("Machine Learning - Classificação de Risco")

    # Carregar e renomear as colunas do DataFrame
    df = pd.read_csv('consolidado.csv')
    df.rename(columns={
        'auto': 'CARRO',
        'moto': 'MOTO',
        'ciclista': 'BICICLETA',
        'ciclom': 'CICLOMOTOR',
        'caminhao': 'CAMINHÃO',
        'pedestre': 'PEDESTRE',
        'outros': 'OUTROS',
        'viatura': 'VIATURA',
        'onibus': 'ÔNIBUS'
    }, inplace=True)

    # Filtrar bairros removendo valores vazios e zeros
    df_filtered = df[df['bairro'].notna()]  # Remove valores NaN
    df_filtered = df_filtered[df_filtered['bairro'] != ""]  # Remove strings vazias
    df_filtered = df_filtered[df_filtered['bairro'] != '0']  # Remove valores '0'

    # Lista única de bairros filtrados
    bairros_unicos = df_filtered['bairro'].unique()

    # Opções de entrada do usuário Bairro
    bairro = st.selectbox("Selecione o bairro:", bairros_unicos)

    # Opções de entrada do usuário Veículo
    veiculos_disponiveis = [col for col in df.columns if col not in ['bairro', 'cidade', 'vitimas', 'vitimasfatais', 'data', 'uf_cidade_bairro', 'Longitude', 'Latitude', 'tipo', 'numero', 'natureza_acidente', 'endereco', 'situacao']]
    veiculo = st.selectbox("Selecione o tipo de veículo:", veiculos_disponiveis)

    # Converta colunas para numérico, forçando erros a NaN
    df_numeric = df.apply(pd.to_numeric, errors='coerce')

    # Função para calcular a classificação com base nos inputs do usuário
    def calcular_classificacao(bairro):
        # Filtrar dados pelo bairro selecionado
        dados_bairro = df_numeric[df_numeric['bairro'] == bairro]
        
        # Calcular métricas
        qtd_vitimas = dados_bairro['vitimas'].sum()
        if 'vitimasfatais' in dados_bairro.columns:
            qtd_vitimas += dados_bairro['vitimasfatais'].sum()

        qtd_ocorrencias = len(dados_bairro)
        
        # Verificar colunas presentes
        colunas_para_excluir = ['bairro', 'cidade', 'vitimas', 'vitimasfatais', 'data', 'uf_cidade_bairro', 'Longitude', 'Latitude', 'tipo', 'numero', 'natureza_acidente', 'endereco', 'situacao']
        colunas_existentes = [col for col in colunas_para_excluir if col in df_numeric.columns]
        
        # Calcular o total de acidentes por tipo de veículo
        total_acidentes = df_numeric.drop(columns=colunas_existentes).sum()
        
        # Classificar o veículo com o maior índice de ocorrência
        if total_acidentes.empty:
            veiculo_com_mais_ocorrencias = None
            maior_ocorrencia = 0
        else:
            veiculo_com_mais_ocorrencias = total_acidentes.idxmax()
            maior_ocorrencia = total_acidentes.max()
        
        # Definir intervalos de classificação
        if qtd_vitimas <= 100 and qtd_ocorrencias <= 100:
            classificacao = "Risco baixo"
            cor_classificacao = "#00FF00"  # Verde
        elif 101 <= qtd_vitimas <= 1000 or 101 <= qtd_ocorrencias <= 1000:
            classificacao = "Risco moderado"
            cor_classificacao = "#FFFF00"  # Amarelo
        else:
            classificacao = "Risco alto"
            cor_classificacao = "#FF0000"  # Vermelho
        
        return classificacao, cor_classificacao, veiculo_com_mais_ocorrencias, maior_ocorrencia

    # Botão para calcular a classificação
    if st.button("Calcular Classificação de Risco"):
        classificacao, cor_classificacao, veiculo_com_mais_ocorrencias, maior_ocorrencia = calcular_classificacao(bairro)
        
        # Defina as cores
        cor_bairro = "#0958D9"  # Azul
        cor_veiculo = "#0958D9"  # Azul
        
        # Crie a string HTML com as cores desejadas
        html_texto = f"""
        <h3>A classificação de risco considerando o bairro de destino, <span style="color:{cor_bairro};">{bairro}</span> , e tipo de veículo utilizado para o deslocamento, <span style="color:{cor_veiculo};">{veiculo}</span> é: <span style="color:{cor_classificacao};"><strong>{classificacao}</strong></span></h3>
        """
        # Renderize o HTML com o Streamlit
        st.markdown(html_texto, unsafe_allow_html=True)

# 10. APLICAÇÃO MERCADOLÓGICA

    st.write("---")
    st.title("Aplicação Mercadológica")
    
    st.write("""
    O dashboard desenvolvido oferece insights valiosos para diversas partes interessadas:

    1. **Órgãos de Trânsito:** Pode ser usado para identificar áreas críticas e implementar medidas de segurança, como sinalização e fiscalização intensiva.
    2. **Seguradoras:** Compreender padrões de acidentes pode ajudar na criação de políticas de seguros mais ajustadas ao risco de cada localidade e perfil de veículo.
    3. **Planejamento Urbano:** Identificar tendências de acidentes pode auxiliar na melhoria do planejamento urbano, promovendo a construção de vias mais seguras e eficientes.
    4. **Empresas de Mobilidade:** Fornece dados para empresas de mobilidade urbana planejarem rotas seguras e eficientes para seus serviços, como aplicativos de transporte ou serviços de entrega.
    5. **Usuários Comuns:** Permite aos cidadãos tomar decisões mais informadas sobre rotas e meios de transporte, aumentando sua segurança.
    """)

    st.write("""
    A aplicação de machine learning integrada pode ser expandida para fornecer recomendações em tempo real, possibilitando ajustes dinâmicos nas rotas com base em condições de trânsito, eventos ou outras variáveis externas. Esta funcionalidade pode ser explorada para desenvolvimentos futuros, integrando sensores de IoT, big data, novas variáveis de diferentes bases de dados, ampliando sua capacidade de análise. Por exemplo, o intervalos de horas do dia com maior índice de acidentes, adotar dados da Secretária de Segurança Pública, na qual poderia ser integrada para incluir métricas de avaliação dos níveis de segurança por localidade, com foco específico em roubos de veículos. Também seria possível incorporar informações do DETRAN, como a evolução do quatitativo da frota de veículos da zona metropolitana do Recife. para previsões mais precisas e ações preventivas.
    """)
   
    st.write("---")
