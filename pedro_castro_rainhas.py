"""
Arquivo principal para a resolução do problema das N Rainhas
Autor: Pedro Augusto de Castro e Castro
Data da última atualização: 25/08/2020
"""
# Importação das bibliotecas necessárias
from random import randint, random, sample, shuffle
from math import floor


def selecao(individuos, pop, N, Cmax):
    """Função de Seleção dos melhores indivíduos"""

    # Verificação da linha - não é necessária pois a representação elimina essa possibilidade
    xeques = [len(individuos[i]) - len(set(individuos[i])) for i in range(0, pop)]

    # Aqui faz a separação indicando qual casa a rainha está. Obs.: tupla da seguinte forma (COLUNA, LINHA)
    pos_rainhas = [list(enumerate(individuos[i], 1)) for i in range(0, pop)]

    for i in range(0, pop):
        for j in range(0, N):
            for c in range(j+1, N):
                # i trabalha com individuos, j trabalha com rainhas dentro de um individuo
                # Caso isso aconteça, ocorrera uma colisao (xeque):
                if abs(pos_rainhas[i][j][0] - pos_rainhas[i][c][0]) == \
                        abs(pos_rainhas[i][j][1] - pos_rainhas[i][c][1]):
                    xeques[i] = xeques[i] + 1

    # Função Fitness
    fitness = [Cmax - xeques[i] for i in range(len(xeques))]

    # Seleção, escolhe os 2 melhores (com menos xeques) dentre todas as opções:
    # range(len(xeques)) indica os indices do vetor
    # key=lambda x: xeques[x] fala para o sorted ordenar de acordo com o valor do elemento do vetor
    melhor = sorted(range(len(fitness)), key=lambda x: fitness[x], reverse=True)[:2]
    # Seleção, escolhe os 2 piores (com mais xeques) dentre todas as opções:
    pior = sorted(range(len(fitness)), key=lambda x: fitness[x])[:2]

    return melhor, pior, fitness


def pedro_castro_rainhas(N):
    """Função principal do problema das N Rainhas """
    # Criação da população inicial:
    tam_pop = 100  # População inicial de 100 indivíduos

    # Quantidade máxima de xeques
    Cmax = floor(N * (N - 1) / 2)

    # A lista de individuos conterá tam_pop individuos.
    # Dessa forma, cada linha da lista sera um individuo (vetor) contendo N valores

    # Criação da lista de individuos iniciais de forma randomica
    individuos = [sample(range(1, N+1), N) for _ in range(0, tam_pop)]

    # Verificar xeques:
    # É preciso verificar os xeques dentro de um mesmo individuo
    num = 0
    while num < 100000:

        # Embaralhar os indíviduos dentro da população
        shuffle(individuos)

        # Fazer a seleção dos individuos com a chamada da função selecao
        melhor, _, fitness = selecao(individuos, tam_pop, N, Cmax)

        if not(Cmax in fitness):
            # selecionados = [individuos[rainha_1], individuos[rainha_2]]
            # print(f'Geração {num+1}: Os xeques: {xeques}')
            pos = randint(0, N-2)  # Posição onde será feito o crossover#

            filho_1 = individuos[melhor[0]][0:pos]
            filho_1 = filho_1 + [individuos[melhor[1]][-i] for i in range(1, N + 1)
                                 if not(individuos[melhor[1]][-i] in individuos[melhor[0]][0:pos])
                                 and len(filho_1) <= N]

            filho_2 = individuos[melhor[1]][0:pos]
            filho_2 = filho_2 + [individuos[melhor[0]][-i] for i in range(1, N + 1)
                                 if not(individuos[melhor[0]][-i] in individuos[melhor[1]][0:pos])
                                 and len(filho_2) <= N]

            p_mut = random()  # probabilidade de ocorrer mutação
            individuos = individuos + [filho_1] + [filho_2]

            if p_mut <= 0.2:  # 20% de probabilidade de ocorrer mutação
                escolha_ind = randint(0, len(individuos)-1)

                escolha_pos_1 = randint(0, N-1)
                escolha_pos_2 = randint(0, N-1)

                aux = individuos[escolha_ind][escolha_pos_1]
                individuos[escolha_ind][escolha_pos_1] = individuos[escolha_ind][escolha_pos_2]
                individuos[escolha_ind][escolha_pos_2] = aux

            shuffle(individuos)

            _, pior, fitness = selecao(individuos, len(individuos), N, Cmax)
            # Para nao eliminar um com o indice correto, e depois eliminar o prox com o indice errado
            if pior[0] > pior[1]:
               individuos.pop(pior[0])
               individuos.pop(pior[1])
            else:
                individuos.pop(pior[1])
                individuos.pop(pior[0])
        else:
            #res = xeques.index(x == 0)
            res = fitness.index(Cmax)

            break
        num = num + 1
    return individuos[res], num 


# Definição do tamanho do tabuleiro e da quantidade de rainhas:
N: int = 8  # Tabuleiro 8x8 com 8 rainhas

# Chamada da função das N rainhas
melhor = pedro_castro_rainhas(N)

print(f'Solução encontrada: {melhor[0]}, número de gerações: {melhor[1]}')
