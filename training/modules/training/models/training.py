import os
import warnings
import pandas as pd
import matplotlib 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pickle
from  tqdm import tqdm
from datetime import datetime
from datetime import date
from sklearn.model_selection import KFold # gerador de data sets de treino e de teste para o modelo de regressão
from sklearn import linear_model # regressor do sklearn
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')
sns.set()

class Training:

    def __init__(self, context):
        self.logging = context.logging

    def apply(self):
        
        # Get data from the excel files
        self.logging.info("Getting data from excel files")
        df_esmat_arts, df_esmat_goepik, dict_df_sag_jandira = self.getData()   

        # Remove files from tmp folder
        temp_folder = "tmp/"
        self.removeFiles([
            temp_folder+"compilado_ARTS.xlsx",
            temp_folder+"eSMAT_Goepik.xlsx",
            temp_folder+"SAG_Jandira.xlsx"])     

        ### TRANSFORMING DATA - ESMAT AND ART'S
        #cleaning data
        df_esmat_arts = self.cleanDfEsmatArts(df_esmat_arts)

        #Getting dict of variables
        dic_esmat_goepik = self.getEsmatGoepikDictionary(df_esmat_goepik)

        #  Replacing 
        df_esmat_goepik.rename(columns= dic_esmat_goepik, inplace=True)

        #setting data and indexing
        df_esmat_goepik['col5']  = pd.to_datetime(df_esmat_goepik['col5']).values.astype('<M8[D]')
        
        #inv_map = {v: k for k, v in dic_esmat_goepik.items()}

        # Creating df critical by local checkpoint2
        df_critic_by_local, s_risk_by_task_art = self.getDfCriticByLocal(df_esmat_goepik, df_esmat_arts)

        # Getting not na
        cond_not_na = ~df_critic_by_local[df_critic_by_local.columns[4]].isna()
        df_critic_by_local_notna = df_critic_by_local[cond_not_na]

        ### TRANSFORMING DATA - SAG, RCI AND SHEERO
        df_aci = dict_df_sag_jandira["Acidentes"][dict_df_sag_jandira["Acidentes"]["Situação do Acidente"] != "CANCELADA"]
        df_aci.drop(df_aci.index.max(), inplace=True)
        df_cond_ins = dict_df_sag_jandira['Condição insegura'][dict_df_sag_jandira['Condição insegura']["Situação do RCI"] != "CANCELADA"]
        df_at_ins = dict_df_sag_jandira['Ato inseguro'][dict_df_sag_jandira['Ato inseguro']["Situação do RCI"] != "CANCELADA"]
        df_at_ins.drop(df_at_ins.index.min(), inplace=True)
        df_qs_aci = dict_df_sag_jandira['QuaseAcidentes'][dict_df_sag_jandira['QuaseAcidentes']["Situação do RCI"] != "CANCELADA"]

        #categorizing
        df_cond_ins['Condição insegura'] = 1 
        df_at_ins['Ato inseguro'] = 1
        df_aci['Acidente'] = 1
        df_qs_aci['Quase acidente'] = 1

        #creating counter
        df_cond_ins.rename(columns={"Registros":"total_cond_ins"}, inplace = True)
        df_at_ins.rename(columns={"Registros":"total_at_ins"}, inplace = True)
        df_aci.rename(columns={"Registros":"total_aci"}, inplace = True)
        df_qs_aci.rename(columns={"Registros":"total_qs_aci"}, inplace = True)

        #
        df_cond_ins =df_cond_ins.loc['2016-05-09':'2019-12-11'] 
        df_at_ins = df_at_ins.loc['2016-05-09':'2019-12-11'] 
        df_qs_aci = df_qs_aci.loc['2016-05-09':'2019-12-11']

        #Join tables
        df_sag_jandira = pd.concat([df_cond_ins, df_at_ins,df_aci,df_qs_aci])

        # Get dictionaries sag to art
        sag_celula_to_art, sag_cipa_to_art, sag_rci_aci_to_art = self.getDictsSagToArt()
                    
        df_sag_jandira_only = df_sag_jandira[df_sag_jandira['Célula'].apply(lambda x: self.select_jand(x))]
        # remap 'célula'
        df_sag_jandira_only.replace({"Local do Acidente": sag_rci_aci_to_art}, inplace = True)
        df_sag_jandira_only.replace({"Local do RCI": sag_rci_aci_to_art}, inplace = True)

        # redefine type (perhaps optional)
        df_sag_jandira_only.astype({'Local do Acidente': 'str', 'Local do RCI':'str'}).dtypes

        # creating column local
        df_sag_jandira_only['Local']  = df_sag_jandira_only.apply(lambda row: row['Local do RCI'] if (str(row['Local do Acidente']) == 'nan') else row['Local do Acidente'] , axis=1)
        
        df_model_sag = df_sag_jandira_only[['Local','total_aci','total_at_ins','total_cond_ins','total_qs_aci','Classificação do Risco']]
        df_model_sag.reset_index(inplace = True)
        df_model_sag.rename(columns={'index':'data'}, inplace = True)
        df_model_sag.dropna(subset=['Local'], inplace=True)
        df_model_sag.fillna(0, inplace = True) 
        df_model_sag_categorized =  self.get_dummie(df_model_sag,'Classificação do Risco')
        df_model_sag_categorized = self.to_numeric(df_model_sag_categorized, ['CRÍTICO', 'LEVE','MODERADO','MUITO CRÍTICO'])

        #grouping by week
        df_model_sag_week = df_model_sag_categorized.groupby('Local').resample('W', on='data').sum().reset_index().sort_values(by='data')
        
        ### JOIN DATA - SAG, ESMAT AND ART
        # JOIN DATA BEFORE GETTING W+3
        # Rename SAG MODEL 
        df_model_sag_week.rename(columns={'CRÍTICO':'sag_critico','LEVE':'sag_leve','MODERADO':'sag_moderado','MUITO CRÍTICO':'sag_muito_critico'}, inplace = True)

        # Rename critic by local
        df_critic_by_local_notna.reset_index(inplace = True)
        df_critic_by_local_notna.columns = ['Local','data','esmat_critico','esmat_leve','esmat_moderado','art_risk_task']

        df_art_esmat_sag = pd.merge(df_critic_by_local_notna, df_model_sag_week, on=['data', 'Local'], how='outer')
        df_art_esmat_sag.set_index('Local', inplace = True)

        # getting art risk by task
        df_art_esmat_sag['art_risk_task'] = s_risk_by_task_art
        # 
        # extatcting 2019 data
        df_art_esmat_sag.reset_index(inplace = True)
        df_art_esmat_sag.set_index('data', inplace = True)
        df_art_esmat_sag_19 = df_art_esmat_sag.loc['2019-02-06':'2019-12-29']
        df_art_esmat_sag_19.fillna(0,inplace = True)

        # from 2019, extracting only art not null data
        cond_zero_art = df_art_esmat_sag_19['art_risk_task'] != 0 
        df_art_esmat_sag_19 = df_art_esmat_sag_19[cond_zero_art].copy()

        #Preparing to the next step
        df_art_esmat_sag_19.reset_index(inplace = True)

        df_art_esmat_sag_19 = self.getACIW3(df_art_esmat_sag_19)

        # ENSURING that only art risked task in model
        list_commun_places = ['academia de ginástica', 'administrativo','almoxarifado de embalagens','carregamento p1','carregamento p2','carregamento p3','central de resíduos','departamento técnico','manutenção','seção 2a','seção 4','seção 5','seção 6','seção 7','seção 8']
        cond_commun_place = df_art_esmat_sag_19['Local'].isin(list_commun_places)
        df_model = df_art_esmat_sag_19[cond_commun_place].copy()

        # dict with commun columnslogging.info
        dict_names_agg = {
        'art_risk_task': 'max',
        'esmat_critico' : 'sum',
        'esmat_leve': 'sum',
        'esmat_moderado': 'sum',
        'sag_critico': 'max',
        'sag_leve': 'max',
        'sag_moderado': 'max',
        'sag_muito_critico': 'max',
        'total_aci': 'max',
        'total_at_ins': 'max',
        'total_cond_ins' : 'max',
        'total_qs_aci' : 'max',
        'aci_w+3' : 'max'}

        # join to cases where esmat was in a diferent line
        df_model = df_model.groupby(['data','Local']).agg(dict_names_agg).reset_index().copy()


        ########################################
        # Modeling
        ########################################

        # getting dummies
        df = self.get_dummie(df_model, 'Local')
        del df['data']

        #################################################
        # TESTE - Apagar este codigo junto com a funcao
        #################################################
        df = self.getTesteDf()

        #################################################
        # TESTE - Fim
        #################################################



        # variável resposta
        #variavel_resposta = 'aci_w+3'

        # features de treinamento
        features = list(set(list(df.columns)) - set('aci_w+3'))

        # features resposta
        output = ['aci_w+3']

        # inputs e output do modelo
        X = df[features]
        y = df[output]

        # Creating model
        lr = LogisticRegression(solver='liblinear')
        lr.fit(X,y)

        return lr
               
        
    def getData(self):
        # Setting paths
        path_separator = '/'
        path_src = 'tmp'

        src1_name_art = 'compilado_ARTS.xlsx' 
        src2_name_goepik = 'eSMAT_Goepik.xlsx'
        workSheets = ['Acidentes','Condição insegura', 'Ato inseguro', 'QuaseAcidentes']
        index_cols = ['Data do Acidente','Data do RCI','Data do RCI','Data do RCI'] 
        src3_name_sag = 'SAG_Jandira.xlsx' 

        #Reading esmat_goepik data src art
        df_esmat_arts = pd.read_excel(path_src + path_separator + src1_name_art)
        df_esmat_goepik = pd.read_excel(path_src + path_separator + src2_name_goepik)

        #Reading sag data
        dict_df_sag_jandira = {}
        for ws, ix in tqdm(zip(workSheets, index_cols)):
            dict_df_sag_jandira[ws] = pd.read_excel(path_src + path_separator + src3_name_sag, sheet_name=ws, index_col=ix, parse_dates=True)

        return df_esmat_arts, df_esmat_goepik, dict_df_sag_jandira

    def removeFiles(self, files):
        for file in files:
            if os.path.exists(file):
                self.logging.info("Removing file " + file)
                os.remove(file)
            else: self.logging.info("File " + file + " doesn't exist!")

    def cleanDfEsmatArts(self, df_esmat_arts):
        df_esmat_arts['Line'] = df_esmat_arts['Line'].apply(lambda x: x.lower())
        df_esmat_arts['Line'] = df_esmat_arts['Line'].apply(lambda x: ' '.join(x.split()))
        df_esmat_arts['Task'] = df_esmat_arts['Task'].apply(lambda x: x.lower())
        df_esmat_arts['Task'] = df_esmat_arts['Task'].apply(lambda x: ' '.join(x.split()))

        return df_esmat_arts

    def getEsmatGoepikDictionary(self, df_esmat_goepik):
        dic_esmat_goepik = {}
        for ix, row in df_esmat_goepik.iterrows():
            if(ix < df_esmat_goepik.shape[1]):
                dic_esmat_goepik[row.index[ix]] = f'col{ix}'
        
        return dic_esmat_goepik

    def getEsmatToArt(self):
        return { 
            'COMPRESSORES' : 'manutenção',
            'LABORATÓRIO' : 'departamento técnico',
            'MANUTENÇÃO': 'manutenção',
            'SALAS DA PRODUÇÃO E ADM' : 'SALAS DA PRODUÇÃO E ADM_esmat_only',
            'VESTIÁRIO' : 'VESTIÁRIO_esmat_only',
            'WEBER BRASIL' : 'WEBER BRASIL_esmat_only',
            'WEBER BRASIL/JANDIRA - SP' : 'WEBER BRASIL/JANDIRA - SP_esmat_only',
            'WEBER BRASIL/JANDIRA - SP/ADM INDUSTRIAL' : 'adm_industrial_esmat_only',
            'WEBER BRASIL/JANDIRA - SP/ALMOXARIFADO' : 'almoxarifado_esmat_only',
            'WEBER BRASIL/JANDIRA - SP/ALMOXARIFADO/EMBALAGENS' : 'almoxarifado de embalagens',
            'WEBER BRASIL/JANDIRA - SP/ALMOXARIFADO/MATERIA PRIMA' : 'almoxarifado materia prima',
            'WEBER BRASIL/JANDIRA - SP/ALMOXARIFADO/PEÇAS' : 'almoxarifado peças',
            'WEBER BRASIL/JANDIRA - SP/EXPEDIÇÃO' : 'expedição_esmat_only',
            'WEBER BRASIL/JANDIRA - SP/EXPEDIÇÃO/CARREGAMENTO P1' : 'carregamento p1',
            'WEBER BRASIL/JANDIRA - SP/EXPEDIÇÃO/CARREGAMENTO P2' : 'carregamento p2',
            'WEBER BRASIL/JANDIRA - SP/EXPEDIÇÃO/CARREGAMENTO P3' : 'carregamento p3',
            'WEBER BRASIL/JANDIRA - SP/EXPEDIÇÃO/PÁTIO DE CARREGAMENTO' : 'PÁTIO DE CARREGAMENTO_esmat_only',
            'WEBER BRASIL/JANDIRA - SP/MANUTENÇÃO' : 'manutenção',
            'WEBER BRASIL/JANDIRA - SP/MANUTENÇÃO/INDUSTRIAL' : 'MANUTENÇÃO/INDUSTRIAL_esmat_only',
            'WEBER BRASIL/JANDIRA - SP/PRODUÇÃO' :'PRODUÇÃO_esmat_only',
            'WEBER BRASIL/JANDIRA - SP/PRODUÇÃO/SEÇÃO 2A' : 'seção 2a',
            'WEBER BRASIL/JANDIRA - SP/PRODUÇÃO/SEÇÃO 2A/DESCARREGAMENTO DE AREIA' : 'seção 2a',
            'WEBER BRASIL/JANDIRA - SP/PRODUÇÃO/SEÇÃO 4' : 'seção 4',
            'WEBER BRASIL/JANDIRA - SP/PRODUÇÃO/SEÇÃO 4/ENSAQUE E PALETIZAÇÃO': 'seção 4',
            'WEBER BRASIL/JANDIRA - SP/PRODUÇÃO/SEÇÃO 5': 'seção 5',
            'WEBER BRASIL/JANDIRA - SP/PRODUÇÃO/SEÇÃO 6': 'seção 6',
            'WEBER BRASIL/JANDIRA - SP/PRODUÇÃO/SEÇÃO 6/ENSAQUE': 'seção 6',
            'WEBER BRASIL/JANDIRA - SP/PRODUÇÃO/SEÇÃO 6/ENSAQUE/MISTURA': 'seção 6',
            'WEBER BRASIL/JANDIRA - SP/PRODUÇÃO/SEÇÃO 6/ENSAQUE/MISTURA/DOSAGEM': 'seção 6',
            'WEBER BRASIL/JANDIRA - SP/PRODUÇÃO/SEÇÃO 7': 'seção 7',
            'WEBER BRASIL/JANDIRA - SP/PRODUÇÃO/SEÇÃO 7/DOSAGEM/PREMIX': 'seção 7',
            'WEBER BRASIL/JANDIRA - SP/PRODUÇÃO/SEÇÃO 7/DOSAGEM/PREMIX/FILTROS': 'seção 7',
            'WEBER BRASIL/JANDIRA - SP/PRODUÇÃO/SEÇÃO 7/MISTURA': 'seção 7',
            'WEBER BRASIL/JANDIRA - SP/PRODUÇÃO/SEÇÃO 8/MONTAGEM E PALETIZAÇÃO' : 'seção 8',
        }

    def getDfCriticByLocal(self, df_esmat_goepik, df_esmat_arts):
        # Converting dictionaries
        esmat_to_art = self.getEsmatToArt()

        columns_item={
            "col0":"id",
            "col14":"atividade",
            "col15":"local",
            'col21':'uso EPI',
            'col24':'condição EPI',
            'col27':'armazenamento EPI',
            'col36':'análise prévia EHS',
            'col39':'compreensão EHS',
            'col42':'consciência riscos EHS',
            'col45':'permissão de trabalho',
            'col48':'bloqueio energias',
            'col51':'participação eventos',
            'col54':'identificação da CI',
            'col57':'postura pró ativa',
            'col66':'ferramentas adequadas',
            'col69':'uso seguro ferramentas',
            'col72':'boa condição ferramentas',
            'col82':'uso correto ergonômicos',
            'col92': 'posição pessoa', #old não tinha essa col92
            'col101':'mudança de EHS',
            'col104':'compromisso em mudar',
        }

        # creating dict item : col_name
        inv_map = {v: k for k, v in columns_item.items()}
        risk_cols = inv_map
        del risk_cols['id']
        del risk_cols['atividade']
        del risk_cols['local']

        b_c_concat = pd.DataFrame()

        for col in risk_cols.values(): 
            a = df_esmat_goepik.groupby(['col15',col,'col5'])[col].agg('count').sort_values(ascending=False).to_frame().rename(columns={col:'qtd'}).unstack([0,1]).resample('W').agg('count') #.stack(1).reset_index() #.fillna(0) #.unstack(level=-1).reset_index()
            b = a.stack(level=1) #.reset_index(inplace = True)
            b.reset_index(inplace = True) #.fillna(0)
            b.columns.names = ['a','B']
            b_c_concat = pd.concat([b_c_concat,b])
            b.reset_index(inplace = True)
            local = b_c_concat.columns[1]
            data = b_c_concat.columns[0]
            df_critic_by_local = b_c_concat.groupby([data,local]).agg('sum') #.set_index(data,local)#.groupby([data,local]).agg('sum')
            #df_critic_by_local = df_teste_data_no_index
            df_critic_by_local.reset_index(inplace=True)
            #df_critic_by_local = b_c_concat.groupby(local).agg('sum')
        df_critic_by_local.set_index(df_critic_by_local.columns[0],inplace = True)
        df_critic_by_local.rename(index= esmat_to_art, inplace = True)

        # calculus risk by task - ART
        s_risk = df_esmat_arts.groupby(['Line'])['Risk Score'].agg('sum')
        s_task = df_esmat_arts.groupby(['Line'])['Task'].agg('count')
        s_risk_by_task_art = s_risk / s_task
        df_critic_by_local['risk_by_task_art'] = s_risk_by_task_art

        # removing duplicate
        df_critic_by_local.drop_duplicates(inplace = True)

        return df_critic_by_local, s_risk_by_task_art

    def getDictsSagToArt(self):
        # creating dict sag to art
        sag_celula_to_art ={
        'JAND ALMOXARIFADO': 'almoxarifado_sag_only',
        'JAND CHEFIA ADM': 'chefia adm_sag_only',
        'JAND CONTROLE DE QUALIDA': 'controle de qualidade_sag_only',
        'JAND EMPILHADEIRA EXP.': 'empilhadeira exp_sag_only',
        'JAND EMPILHADEIRA PRD.': 'empilhadeira prd_sag_only',
        'JAND EXPEDIÇÃO': 'expedição_sag_only',
        'JAND GERENCIA INDUSTRIAL': 'gerencia industrial_sag_only',
        'JAND MANUTENÇÃO': 'manutenção',
        'JAND PROD. LINHA 04 ARGA': 'seção 4',
        'JAND PROD. LINHA 05 ARGA': 'seção 5',
        'JAND PROD. LINHA 07 ARGA': 'seção 7',
        'JAND PROD. LINHA 08 ARGA': 'seção 8',
        'JAND PROD. LINHA 2A REJ': 'seção 2a',
        'JAND PROD. LINHA PREMIX': 'linha premix_sag_only'}

        sag_cipa_to_art ={
        'ADMINISTRATIVO / FINANCEIRO': 'administrativo',
        'CARREGAMENTO / EXPEDIÇÃO': 'carregamento/exped_sag_only',
        'DEPARTAMENTO TÉCNICO':'departamento técnico',
        'INDUSTRIAL ADM' : 'industrial_adm_sag_only',
        'MANUTENÇÃO':'manutenção',
        'PREMIX / ALMOXARIFADO': 'premix/almoxarifado_sag_only',
        'RECURSOS HUMANOS': 'rh_sag_only',
        'SEÇÃO 2A':'seção 2a',
        'SEÇÃO 4': 'seção 4',
        'SEÇÃO 5': 'seção 5',
        'SEÇÃO 7': 'seção 7'}

        sag_rci_aci_to_art = {
        'ACADEMIA DE FACHADAS': 'academia_fachada_sag_only',
        'ACADEMIA DE GINASTICA' : 'academia de ginástica',
        'ADM INDUSTRIAL' : 'administrativo',
        'ADMINISTRATIVO' : 'administrativo',
        'ALMOXARIFADO' : 'almoxarifado_sag_only',
        'AREA DE CAFE' : 'area_de_cafe_sag_only',
        'AREAS COMUM' : 'areas_comum_sag_only',
        'ATENDIMENTO' : 'atendimento_sag_only',
        'BANHEIROS' : 'banheiros_sag_only',
        'BLOCO C' : 'bloco_c_sag_only' ,
        'CARREGAMENTO P1' : 'carregamento p1',
        'CARREGAMENTO P2': 'carregamento p2',
        'CARREGAMENTO P3': 'carregamento p3',
        'CENTRAL DE RESIDUOS' : 'central de resíduos',
        'CENTRO MÉDICO' : 'centro_médico_sag_only',
        'CHEFIA ADM' : 'chefia_adm_sag_only',
        'COLETORES (RESIDUO)' : 'central de resíduos',
        'COMERCIAL / VENDAS' : 'comercial_vendas_sag_only',
        'COMERCIAL INTERNO' : 'comercial_interno_sag_only',
        'COMUNICAÇÃO': 'comunicacao_sag_only',
        'CONTROLE QUALID': 'controle_e_qualidade_sag_only',
        'CORP. INDUSTRIAL' : 'corp_industrial_sag_only',
        'DESCARREGAMENTO DE AREIA': 'seção 5',
        'DESENVOLVIMENTO DE PROD' : 'desn_prod_sag_only',
        'DIRETORIA' : 'administrativo',
        'DOSAGEM': 'seção 2a',
        'DT' : 'administrativo',
        'EHS / SGI' : 'administrativo',
        'EHS/WCM/SGI CORPORATIVO' : 'administrativo',
        'EMBALAGENS' : 'almoxarifado de embalagens',
        'ENGENHARIA/PROJETOS' : 'administrativo',
        'ENSACADEIRA' : 'ensacadeira_sag_only',
        'ENSAQUE' : 'ensaque_sag_only',
        'ENSAQUE E PALETIZAÇÃO' : 'ensaque_e_paletização_sag_only',
        'ESCADA / ACESSO RECEPÇÃO' : 'escada_recepção_sag_only',
        'ESCADAS E ÁREAS DE ACESSO' : 'escadas_e_acesso_sag_only',
        'ESTACIONAMENTO' : 'estacionamento_sag_only',
        'EXPEDIÇÃO' : 'expedição_sag_only',
        'FATURAMENTO' : 'faturamento_sag_only',
        'FILTROS' : 'filtros_sag_only',
        'FINANCEIRO' :'financeiro_sag_only',
        'GERENCIA IND' : 'administrativo',
        'INDUSTRIAL' : 'industrial_sag_only',
        'JANDIRA - SP' : 'jandira_sp_sag_only',
        'LOGISTICA' : 'logistica_sag_only',
        'MANUTENÇÃO' : 'manutenção',
        'MARKETING' : 'administrativo',
        'MATERIA PRIMA' : 'materia_prima',
        'MISTURA' : 'mistura_sag_only',
        'MOEGA' : 'moega_sag_only',
        'PALETIZAÇÃO' : 'paletização_sag_only',
        'PASSARELA' : 'passarela_sag_only',
        'PATRIMONIAL' : 'patrimonial_sag_only',
        'PEÇAS': 'peças_sag_only',
        'PREMIX' : 'premix_sag_only',
        'PRODUÇÃO' : 'produção_sag_only',
        'PÁTIO DE CARREGAMENTO' : 'patio_carregamento_sag_only',
        'QUIOSQUE' : 'quiosque_sag_only',
        'RECEPÇÃO' : 'recepção_sag_only',
        'RESTAURANTE' : 'restaurante_sag_only',
        'RH' : 'administrativo',
        'SALA DE REUNIÃO' : 'sala_reunião_sag_only',
        'SALAO DE JOGOS': 'sala_de_jogos_sag_only',
        'SEÇÃO 2A': 'seção 2a',
        'SEÇÃO 4': 'seção 4',
        'SEÇÃO 5': 'seção 5',
        'SEÇÃO 6' : 'seção 6',
        'SEÇÃO 7' : 'seção 7',
        'SEÇÃO 8' : 'seção 8',
        'SUPPLY CHAIN': 'supply_chain_sag_only',
        'SUPRIMENTOS' : 'suprimento_sag_only',
        'UTILIDADES' : 'utildades_sag_only',
        'VAN DE TRANSPORTE' : 'van_transporte_sag_only',
        'WCM CORP' : 'administrativo',
        'GEOLOGIA' : 'geologia_sag_only'
        }

        return sag_celula_to_art, sag_cipa_to_art, sag_rci_aci_to_art

    def select_jand(self, str_local):        
        ''' 
        Select only sag-Jandira
        '''
        
        if (str_local.split()[0] == 'JAND'):
            return True
        else:
            return False

    # return di=ummied data frame by categorical input
    def get_dummie(self, df, categorical):
        
        dummie = pd.get_dummies(df[categorical])
        df = pd.concat([df,dummie], axis = 1)
        del df[categorical]
        return df

    #set to numeric

    def to_numeric(self, df, list_cols):
        for col in list_cols:
            df[col] = df[col].astype(float)     

        return df

    def getACIW3(self, df_art_esmat_sag_19):
        # Getting w+3 aci
        areas = set(df_art_esmat_sag_19['Local'])
        for area in tqdm(areas):
            cond_area = df_art_esmat_sag_19['Local'] == area
            list_ix = df_art_esmat_sag_19[cond_area].sort_values(by=['data']).index
            for ix in list_ix:
                if (ix < max(list_ix) - 2):
                    sum_3week = df_art_esmat_sag_19.loc[ix + 1,'total_aci'] + df_art_esmat_sag_19.loc[ix + 2,'total_aci'] + df_art_esmat_sag_19.loc[ix + 3,'total_aci'] 
                    if (sum_3week > 0):
                        df_art_esmat_sag_19.loc[ix,'aci_w+3'] = 1
                    else:
                        df_art_esmat_sag_19.loc[ix,'aci_w+3'] = 0#sum_3week - df_model_sag_week.loc[ix,'total_aci']
                elif (ix < max(list_ix) - 1):
                    sum_3week = df_art_esmat_sag_19.loc[ix + 1,'total_aci'] + df_art_esmat_sag_19.loc[ix + 2,'total_aci'] #+ df_model_sag_week.loc[ix + 3,'total_aci'] 
                    if (sum_3week > 0):
                        df_art_esmat_sag_19.loc[ix,'aci_w+3'] = 1
                    else:
                        df_art_esmat_sag_19.loc[ix,'aci_w+3'] = 0
                elif (ix < max(list_ix)):
                    sum_3week = df_art_esmat_sag_19.loc[ix + 1,'total_aci'] #+ df_model_sag_week.loc[ix + 2,'total_aci'] #+ df_model_sag_week.loc[ix + 3,'total_aci'] 
                    if (sum_3week > 0):
                        df_art_esmat_sag_19.loc[ix,'aci_w+3'] = 1
                    else:
                        df_art_esmat_sag_19.loc[ix,'aci_w+3'] = 0
                else:
                    df_art_esmat_sag_19.loc[ix,'aci_w+3'] = 0
        
        return df_art_esmat_sag_19

    # TESTE - Apagar essa funcao
    def getTesteDf(self):
        path_src = "tmp"
        
        df_xunxo = pd.read_excel(path_src + '/' +'model_PRE_XUNXO_3_3_9_50.xlsx', encoding='utf-8')

        df_art_esmat_sag_19 = df_xunxo.copy()

        # ENSURING that only art risked task in model
        list_commun_places = ['academia de ginástica', 'administrativo','almoxarifado de embalagens','carregamento p1','carregamento p2','carregamento p3','central de resíduos','departamento técnico','manutenção','seção 2a','seção 4','seção 5','seção 6','seção 7','seção 8']
        cond_commun_place = df_art_esmat_sag_19['Local'].isin(list_commun_places)
        df_model = df_art_esmat_sag_19[cond_commun_place].copy()



        # dict with commun columns
        dict_names_agg = {
        'art_risk_task': 'max',
        'esmat_critico' : 'sum',
        'esmat_leve': 'sum',
        'esmat_moderado': 'sum',
        'sag_critico': 'max',
        'sag_leve': 'max',
        'sag_moderado': 'max',
        'sag_muito_critico': 'max',
        'total_aci': 'max',
        'total_at_ins': 'max',
        'total_cond_ins' : 'max',
        'total_qs_aci' : 'max',
        'aci_w+3' : 'max'}

        # join to cases where esmat was in a diferent line
        df_model = df_model.groupby(['data','Local']).agg(dict_names_agg).reset_index().copy()

        # getting dummies
        df = self.get_dummie(df_model, 'Local')
        del df['data']

        return df