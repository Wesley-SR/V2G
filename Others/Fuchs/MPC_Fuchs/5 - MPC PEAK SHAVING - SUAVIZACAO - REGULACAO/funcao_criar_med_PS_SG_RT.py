def criar_medicoes_PS_SG_RT(janela,N):
    import csv
    #---------- ESCREVER O ARQUIVO COM DADOS DE MEDIÇÃO -------------------------#

    #------------------------ FORMATO DO ARQUIVO --------------------------------#
    # 'tempo' , 'potencia', 'potencia_PV', 'custo_energia'
    #    0    ,   x [kW]  ,     y[kW]    ,    z[$/kWh]
    #    - - - - - - - - - - - - - - - - - - - - - - -
    #   143   ,   x [kW]  ,     y[kW]    ,    z[$/kWh]
    # ---------------------------------------------------------------------------#

    dados_medicoes = open('medicoes_PS_SG_RT.csv', 'w')
    writer = csv.writer(dados_medicoes, lineterminator='\n')
    writer.writerow(('tempo', 'potencia_ativa', 'potencia_reativa', 'potencia_PV', 'custo_energia'))
    u = 0
    for l in range(0, janela):
        N[l][0] = u
        writer.writerow((N[l][0], N[l][1], N[l][2], N[l][3], N[l][4]))
        u = u + 1
    dados_medicoes.close()

