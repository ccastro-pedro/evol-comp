"""
Algoritmo genético testado com a função de benchmarking de Rastrigin.

Autor: Pedro Augusto de Castro e Castro

Última atualização: 26/08/2020
"""
from random import randint, random, sample
from math import pi, cos, floor
import time
import matplotlib.pyplot as plt


def grayToBinary(g, nvar, li, tam_pop):
    """Faz a conversão de gray para binario"""
    b = [[[0] * li] * nvar] * tam_pop
    for var in range(tam_pop):
        b[var] = g[var].copy()
        for i in range(nvar):
            b[var][i] = g[var][i].copy()
            for j in range(1, li):
                b[var][i][j] = b[var][i][j - 1] ^ g[var][i][j]
    return b


def grayToReal(g, nvar, li, tam_pop, Li, Ui):
    """Faz a conversão de gray para real"""
    # Passo 1: converter gray para binário
    binario = grayToBinary(g, nvar, li, tam_pop)
    # Passo 2: converter de binario para decimal
    ki = [[0] * nvar] * tam_pop
    for var in range(tam_pop):
        ki[var] = ki[var].copy()
        for i in range(nvar):
            for j in range(li):
                ki[var][i] = ki[var][i] + binario[var][i][j] * 2 ** (li - (j + 1))
    # Passo 3: converter de decimal para real
    xi = [[Li + (Ui - Li) * (ki[j][i] / (2 ** li - 1)) for i in range(nvar)] for j in range(tam_pop)]
    return binario, ki, xi


def Fitness(xi, tam_pop, nvar):
    """Função para o cálculo do fitness"""
    Cmax = 200
    aux = [0] * tam_pop
    for j in range(tam_pop):
        for i in range(nvar):
            aux[j] = aux[j] + (xi[j][i] ** 2 - 10 * cos(2 * pi * xi[j][i]))
    fx = [10 * nvar + aux[i] for i in range(tam_pop)]
    fitness_xi = [Cmax - fx[i] if fx[i] < Cmax else 0 for i in range(len(fx))]
    return fitness_xi


def Roleta(fitness, tam_pop, s_rank, qtd_filhos):
    """
    Função para fazer a seleção pelo metodo da roleta. Calcula as probabilidas de seleção, a partir do fitness
    retorna os indices dos pais escolhidos
    """
    # Como são selecionados 2 filhos por pai de pais, é preciso que qtd_filhos seja sempre multiplo de 2
    if qtd_filhos % 2 == 0:
        qtd_filhos = qtd_filhos
    else:
        qtd_filhos = qtd_filhos + 1
    # Operador responsavel por definir as probabilidades de seleção PSi. Foi utilizado o ranking linear
    rank = sorted(range(len(fitness)), key=lambda k: fitness[k], reverse=True)
    # lista de tuplas: (indice, probabilidade)
    Plin_rank = [(rank[i], (2 - s_rank) / tam_pop + (2 * (tam_pop - 1 - i) * (s_rank - 1)) / (tam_pop * (tam_pop - 1)))
                 for i in range(tam_pop)]
    segmento_reta = [0] * tam_pop
    for i in range(tam_pop):
        if i == 0:
            segmento_reta[i] = Plin_rank[::-1][i][1]
        else:
            segmento_reta[i] = segmento_reta[i - 1] + Plin_rank[::-1][i][1]
    segmento_reta_rank = [(segmento_reta[i], rank[::-1][i]) for i in range(tam_pop)]
    pais = [0] * qtd_filhos
    index = 0
    while index <= qtd_filhos - 1:
        r_aleatorio = random()
        for i in range(tam_pop):
            if i == 0:
                if r_aleatorio < segmento_reta_rank[i][0]:
                    pais[index] = segmento_reta_rank[i][1]
            elif segmento_reta_rank[i - 1][0] <= r_aleatorio < segmento_reta_rank[i][0]:
                pais[index] = segmento_reta_rank[i][1]
        index = index + 1
    return pais


def Torneio(xi, k_torneio, qtd_filhos, nvar):
    """Função para seleção pelo metodo de torneio"""
    if qtd_filhos % 2 == 0:
        qtd_filhos = qtd_filhos
    else:
        qtd_filhos = qtd_filhos + 1

    pais = [None] * qtd_filhos
    piores = [None] * qtd_filhos
    i = 0
    while i <= qtd_filhos-1:
        selec_index = sample(range(len(xi)), k_torneio)
        xi_torneio = [xi[j] for j in selec_index]
        fitness_xi_torneio = Fitness(xi_torneio, k_torneio, nvar)
        max_index = fitness_xi_torneio.index(max(fitness_xi_torneio))
        if any(v is None for v in pais):
            pais[i] = selec_index[max_index]
        min_index = fitness_xi_torneio.index(min(fitness_xi_torneio))
        if not(selec_index[min_index] in piores):
            piores[i] = selec_index[min_index]
            i = i + 1
    return pais, piores


def Crossover(individuos_x, pais, nvar, li):
    """Função de Crossover com ponto de corte aleatórios"""
    pontos_corte = [[randint(1, li - 1) for _ in range(nvar)] for _ in range(floor(len(pais) / 2))]
    filhos = [[[None] * li] * nvar] * len(pais)
    for j in range(0, len(pais), 2):
        filhos[j] = filhos[j].copy()
        filhos[j + 1] = filhos[j + 1].copy()
        for i in range(nvar):
            filhos[j][i] = individuos_x[pais[j]][i][0:pontos_corte[floor(j / 2)][i]] + \
                           individuos_x[pais[j + 1]][i][pontos_corte[floor(j / 2)][i]:li]
            filhos[j + 1][i] = individuos_x[pais[j + 1]][i][0:pontos_corte[floor(j / 2)][i]] + \
                               individuos_x[pais[j]][i][pontos_corte[floor(j / 2)][i]:li]
    return filhos


def Mutation(filhos, nvar, li, pm):
    """Função de mutação do tipo bit-flip"""
    for k in range(len(filhos)):
        for j in range(nvar):
            for i in range(li):
                prob_mut = random()
                if prob_mut <= pm/2:
                    if filhos[k][j][i] == 0:
                        filhos[k][j][i] = 1
                    else:
                        filhos[k][j][i] = 0
    return filhos


def pedro_de_castro(nvar, ncal):
    # Numero de bits utilizado para codificar a variavel xi
    li = 5
    # Tamanho da população
    tam_pop = 10
    # Qtd de vezes que o codigo ficou preso em um minimo local
    preso = 0
    # Qtd de vezes que o codigo pode ficar preso em um minimo local
    cal_max = 100
    # Numero de filhos a ser gerado
    qtd_filhos = 4
    # Tolerancia para verificação de minimos locais:
    tol = 5
    # Limites inferior e superior da variavel xi
    Li, Ui = -5.12, 5.12
    # Probabilidade de ocorrer crossover (pc) e mutação (pm)
    pc, pm_init = 0.9, 0.2
    # Probabilidade de ocorrer roleta ou torneio (isso é avaliado depois de pc)
    p_roleta = 0.5
    p_torneio = 1 - p_roleta
    # Numero de individuos no torneio
    k_torneio = floor(tam_pop/2)
    # Inclinação da curva para ranking linear
    s_rank = 1.5
    # Lista com nvar variaveis de tamanho li representando a população inicial
    # Notação em gray
    individuos_x = [[[randint(0, 1) for _ in range(li)] for _ in range(nvar)] for _ in range(tam_pop)]
    cal = 0
    while cal <= ncal:
        _, _, xi = grayToReal(individuos_x, nvar, li, tam_pop, Li, Ui)
        fitness_xi = Fitness(xi, tam_pop, nvar)
        prob_sel = random()
        if prob_sel < p_roleta:
            pais = Roleta(fitness_xi, tam_pop, s_rank, qtd_filhos)
        elif p_roleta <= prob_sel <= p_roleta + p_torneio:
            pais, _ = Torneio(xi, k_torneio, qtd_filhos, nvar)
        prob_cross = random()
        if prob_cross <= pc:
            filhos = Crossover(individuos_x, pais, nvar, li)
        else:
            filhos = [individuos_x[pais[i]] for i in range(len(pais))]
        prob_mut = random()
        # Probabilidade de mutação de acordo com o numero de vezes preso num minimo local
        if pm_init < preso / cal_max:
            pm = preso / cal_max
        else:
            pm = pm_init
        if prob_mut <= pm:
            filhos = Mutation(filhos, nvar, li, pm)
        individuos_x = individuos_x + filhos
        _, piores = Torneio(xi, k_torneio, qtd_filhos, nvar)
        for index in sorted(piores, reverse=True):
            del individuos_x[index]
        binario, ki, xi = grayToReal(individuos_x, nvar, li, tam_pop, Li, Ui)
        fit = Fitness(xi, tam_pop, nvar)
        melhor = fit.index(max(fit))
        if cal == 0:
            mais_adaptado = [individuos_x[melhor]]
            _, _, xi_mais_adaptado = grayToReal(mais_adaptado, nvar, li, 1, Li, Ui)
        else:
            if max(Fitness(xi_mais_adaptado, 1, nvar)) - tol <= max(fit) \
                    <= max(Fitness(xi_mais_adaptado, 1, nvar)) - tol:
                preso = preso + 1
            elif max(fit) >= max(Fitness(xi_mais_adaptado, 1, nvar)):
                mais_adaptado = [individuos_x[melhor]]
                _, _, xi_mais_adaptado = grayToReal(mais_adaptado, nvar, li, 1, Li, Ui)
                preso = 0 # zera o preso pq o codigo saiu da faixa
        cal = cal + 1
    binario, ki, xi = grayToReal(individuos_x, nvar, li, tam_pop, Li, Ui)
    aux = [0] * tam_pop
    for j in range(tam_pop):
        for i in range(nvar):
            aux[j] = aux[j] + (xi[j][i] ** 2 - 10 * cos(2 * pi * xi[j][i]))
    fx = [10 * nvar + aux[i] for i in range(tam_pop)]
    min_index = fx.index(min(fx))
    return xi[min_index], fx[min_index]


# Numero total de chamadas da funçao de calculo
ncal = 100000

# Numero de variavei utilizado
nvar = 10

start_time = time.time()
x, fx = pedro_de_castro(nvar, ncal)
print(f'O programa de otimização levou {time.time() - start_time} s')
print(f'O melhor vetor encontrado foi: {x}')
print(f'O valor da função de Rastrigin desse vetor é: {fx}')