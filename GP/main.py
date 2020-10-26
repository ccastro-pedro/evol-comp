from random import  random, randint, seed
import math
from copy import deepcopy
from statistics import mean, stdev
from printBTree import printBTree
import inspect
import functools as fn
import os
import matplotlib.pyplot as plt

POP_SIZE        = 20   # population size
MIN_DEPTH       = 2    # minimal initial random tree depth
MAX_DEPTH       = 4    # maximal initial random tree depth
GENERATIONS     = 1000  # maximal number of generations to run evolution
TOURNAMENT_SIZE = 5    # size of tournament for tournament selection
XO_RATE         = 0.8  # crossover rate
PROB_MUTATION   = 0.2  # per-node mutation probability

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

# Exemplo de GP para resolver uma simples equação matemática:

def plus(x, y): return x + y


def minus(x, y): return x - y


def times(x, y): return x * y


def division(x, y):
    if y == 0:
        return 1
    else:
        return x / y

def lg(x):
    if x < 0:
        return 1
    else:
        return math.log(x)


def goal(x, y=0): return + x + 1
# return 2 * x + 1
# return x ** 2 + x + 1
# return x ** 3 + x ** 1 + 2
# return x*x*x*x + x*x*x + x*x + x + 1
# return y * x + 5 * y + x
# return x ** 2 + y * x + 5
# return x ** 4 + x ** 2 + 1
# return y ** 2 + x + 5
# return y ** 3 + x ** 2 + x * y + 18
# return 4 * x ** 3 + 5 * x ** 1 + 18

def generate_dataset():
    dataset = []
    for y in drange(-5, 5, 0.1):
        for x in drange(-5, 5, 0.1):
            dataset.append([x, y, goal(x, y)])
    return dataset

def fitness(individual, dataset):
    dif = [abs(individual.compute_tree(ds[0], ds[1]) - ds[2]) for ds in dataset]
    MAE = sum(dif) / len(dif)
    NMAE = 1 / (1 + MAE)
    return NMAE



class GPTree:
    def __init__(self, val=None, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def node_label(self):  # string label
        if self.val in function_set:
            return self.val.__name__
        else:
            return str(self.val)

    def write_txt(self, goal, gen, fit, nodeInfo=lambda n: (n.node_label(), n.left, n.right), inverted=False, isTop=True):
        lines = inspect.getsource(goal)
        lines = lines.split('return')[1].lstrip().rstrip()
        i = 0
        while os.path.exists("GP%s.txt" % i):
            i += 1
        with open("GP%s.txt" % i, "w+") as file:
            file.write(f"Função objetivo: {lines} \nNúmero de gerações: {gen} \nFitness: {fit}")
            file.write("\n\n\n ------------------ \n\n\n")
            # node value string and sub nodes
            stringValue, leftNode, rightNode = nodeInfo(self)

            stringValueWidth = len(stringValue)

            # recurse to sub nodes to obtain line blocks on left and right
            leftTextBlock = [] if not leftNode else printBTree(leftNode, nodeInfo, inverted, False)

            rightTextBlock = [] if not rightNode else printBTree(rightNode, nodeInfo, inverted, False)

            # count common and maximum number of sub node lines
            commonLines = min(len(leftTextBlock), len(rightTextBlock))
            subLevelLines = max(len(rightTextBlock), len(leftTextBlock))

            # extend lines on shallower side to get same number of lines on both sides
            leftSubLines = leftTextBlock + [""] * (subLevelLines - len(leftTextBlock))
            rightSubLines = rightTextBlock + [""] * (subLevelLines - len(rightTextBlock))

            # compute location of value or link bar for all left and right sub nodes
            #   * left node's value ends at line's width
            #   * right node's value starts after initial spaces
            leftLineWidths = [len(line) for line in leftSubLines]
            rightLineIndents = [len(line) - len(line.lstrip(" ")) for line in rightSubLines]

            # top line value locations, will be used to determine position of current node & link bars
            firstLeftWidth = (leftLineWidths + [0])[0]
            firstRightIndent = (rightLineIndents + [0])[0]

            # width of sub node link under node value (i.e. with slashes if any)
            # aims to center link bars under the value if value is wide enough
            #
            # ValueLine:    v     vv    vvvvvv   vvvvv
            # LinkLine:    / \   /  \    /  \     / \
            #
            linkSpacing = min(stringValueWidth, 2 - stringValueWidth % 2)
            leftLinkBar = 1 if leftNode else 0
            rightLinkBar = 1 if rightNode else 0
            minLinkWidth = leftLinkBar + linkSpacing + rightLinkBar
            valueOffset = (stringValueWidth - linkSpacing) // 2

            # find optimal position for right side top node
            #   * must allow room for link bars above and between left and right top nodes
            #   * must not overlap lower level nodes on any given line (allow gap of minSpacing)
            #   * can be offset to the left if lower subNodes of right node
            #     have no overlap with subNodes of left node
            minSpacing = 2
            rightNodePosition = fn.reduce(lambda r, i: max(r, i[0] + minSpacing + firstRightIndent - i[1]), \
                                          zip(leftLineWidths, rightLineIndents[0:commonLines]), \
                                          firstLeftWidth + minLinkWidth)

            # extend basic link bars (slashes) with underlines to reach left and right
            # top nodes.
            #
            #        vvvvv
            #       __/ \__
            #      L       R
            #
            linkExtraWidth = max(0, rightNodePosition - firstLeftWidth - minLinkWidth)
            rightLinkExtra = linkExtraWidth // 2
            leftLinkExtra = linkExtraWidth - rightLinkExtra

            # build value line taking into account left indent and link bar extension (on left side)
            valueIndent = max(0, firstLeftWidth + leftLinkExtra + leftLinkBar - valueOffset)
            valueLine = " " * max(0, valueIndent) + stringValue
            slash = "\\" if inverted else "/"
            backslash = "/" if inverted else "\\"
            uLine = "¯" if inverted else "_"

            # build left side of link line
            leftLink = "" if not leftNode else (" " * firstLeftWidth + uLine * leftLinkExtra + slash)

            # build right side of link line (includes blank spaces under top node value)
            rightLinkOffset = linkSpacing + valueOffset * (1 - leftLinkBar)
            rightLink = "" if not rightNode else (" " * rightLinkOffset + backslash + uLine * rightLinkExtra)

            # full link line (will be empty if there are no sub nodes)
            linkLine = leftLink + rightLink

            # will need to offset left side lines if right side sub nodes extend beyond left margin
            # can happen if left subtree is shorter (in height) than right side subtree
            leftIndentWidth = max(0, firstRightIndent - rightNodePosition)
            leftIndent = " " * leftIndentWidth
            indentedLeftLines = [(leftIndent if line else "") + line for line in leftSubLines]

            # compute distance between left and right sublines based on their value position
            # can be negative if leading spaces need to be removed from right side
            mergeOffsets = [len(line) for line in indentedLeftLines]
            mergeOffsets = [leftIndentWidth + rightNodePosition - firstRightIndent - w for w in mergeOffsets]
            mergeOffsets = [p if rightSubLines[i] else 0 for i, p in enumerate(mergeOffsets)]

            # combine left and right lines using computed offsets
            #   * indented left sub lines
            #   * spaces between left and right lines
            #   * right sub line with extra leading blanks removed.
            mergedSubLines = zip(range(len(mergeOffsets)), mergeOffsets, indentedLeftLines)
            mergedSubLines = [(i, p, line + (" " * max(0, p))) for i, p, line in mergedSubLines]
            mergedSubLines = [line + rightSubLines[i][max(0, -p):] for i, p, line in mergedSubLines]

            # Assemble final result combining
            #  * node value string
            #  * link line (if any)
            #  * merged lines from left and right sub trees (if any)
            treeLines = [leftIndent + valueLine] + ([] if not linkLine else [leftIndent + linkLine]) + mergedSubLines

            # invert final result if requested
            treeLines = reversed(treeLines) if inverted and isTop else treeLines

            # return intermediate tree lines or print final result
            if isTop:
                file.write("\n".join(treeLines))
            else:
                return treeLines
            file.write("\n\n\n----------------------\n\n\n")

    def print_tree(self):
        printBTree(self, lambda n: (n.node_label(), n.left, n.right))

    def compute_tree(self, x, y=0):
        if self.val in function_set:
            return self.val(self.left.compute_tree(x, y), self.right.compute_tree(x, y))
        elif self.val == 'x':
            return x
        elif self.val == 'y':
            return y
        else:
            return self.val

    def random_tree(self, grow, max_depth, depth=0):
        if depth < MIN_DEPTH or (depth < max_depth and not grow):
            self.val = function_set[randint(0, len(function_set) - 1)]
        elif depth >= max_depth:
            self.val = terminal_set[randint(0, len(terminal_set) - 1)]
        else:
            if random() > 0.5:
                self.val = terminal_set[randint(0, len(terminal_set) - 1)]
            else:
                self.val = function_set[randint(0, len(function_set) - 1)]
        if self.val in function_set:
            self.left = GPTree()
            self.left.random_tree(grow, max_depth, depth=depth + 1)
            self.right = GPTree()
            self.right.random_tree(grow, max_depth, depth=depth + 1)

    def mutation(self):
        if random() < PROB_MUTATION:
            self.random_tree(grow=True, max_depth=2)
        elif self.left:
            self.left.mutation()
        elif self.right:
            self.right.mutation()

    def size(self):
        if self.val in terminal_set: return 1
        l = self.left.size() if self.left else 0
        r = self.right.size() if self.right else 0
        return 1 + l + r

    def build_subtree(self):
        t = GPTree()
        t.val = self.val
        if self.left:  t.left = self.left.build_subtree()
        if self.right: t.right = self.right.build_subtree()
        return t

    def scan_tree(self, count, second):
        count[0] -= 1
        if count[0] <= 1:
            if not second:
                return self.build_subtree()
            else:
                self.val = second.val
                self.left = second.left
                self.right = second.right
        else:
            ret = None
            if self.left and count[0] > 1: ret = self.left.scan_tree(count, second)
            if self.right  and count[0] > 1: ret = self.right.scan_tree(count, second)
            return ret

    def crossover(self, other):
        if random() < XO_RATE:
            second = other.scan_tree([randint(1, other.size())], None)
            self.scan_tree([randint(1, self.size())], second)


def init_population():
    population = []
    for md in range(3, MAX_DEPTH + 1):
        for i in range(int(POP_SIZE/2)):
            t = GPTree()
            t.random_tree(grow=True, max_depth=md)  # grow
            population.append(t)
        for i in range(int(POP_SIZE/2)):
            t = GPTree()
            t.random_tree(grow=False, max_depth=md)  # full
            population.append(t)
    return population

# Fitness varia de 0 a 1, tenho que analisar o tamanho dos indivíduos nessa faixa, para definir o melhor
def selection(population, dataset):
    fitnesses = [fitness(ind, dataset) for ind in population]
    tournament = [randint(0, len(population)-1) for _ in range(TOURNAMENT_SIZE)]
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(TOURNAMENT_SIZE)]
    tournament_size = [population[tournament[i]].size() for i in range(TOURNAMENT_SIZE)]
    # Definir o peso para o tamanho dos individuos
    tournament_values = [tournament_fitnesses[i] + 0.1 / tournament_size[i] for i in range(TOURNAMENT_SIZE)]
    return deepcopy(population[tournament[tournament_values.index(max(tournament_values))]])
    #return deepcopy(population[tournament[tournament_fitnesses.index(max(tournament_fitnesses))]])


def remove_selection(population, dataset):
    fitnesses = [fitness(ind, dataset) for ind in population]
    tournament = [randint(0, len(population)-1) for _ in range(TOURNAMENT_SIZE)]
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(TOURNAMENT_SIZE)]
    tournament_size = [population[tournament[i]].size() for i in range(TOURNAMENT_SIZE)]
    # Definir o peso para o tamanho dos individuos
    tournament_values = [tournament_fitnesses[i] + 0 / tournament_size[i] for i in range(TOURNAMENT_SIZE)]
    return population.pop(tournament[tournament_values.index(min(tournament_values))])

def main():
    seed()
    dataset = generate_dataset()
    population = init_population()
    best_of_run = None
    best_of_run_f = 0
    best_of_run_gen = 0
    for gen in range(GENERATIONS):
        nextgen_population = []
        fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]
        best_before = population[fitnesses.index(max(fitnesses))]
        best_before_f = fitnesses[fitnesses.index(max(fitnesses))]
        for i in range(POP_SIZE):
            parent1 = selection(population, dataset)
            parent2 = selection(population, dataset)
            parent1.crossover(parent2)
            parent1.mutation()
            nextgen_population.append(parent1)
        fitnesses = [fitness(nextgen_population[i], dataset) for i in range(POP_SIZE)]
        best_after = nextgen_population[fitnesses.index(max(fitnesses))]
        best_after_f = fitnesses[fitnesses.index(max(fitnesses))]
        if best_before_f > best_after_f:
            remove_selection(nextgen_population, dataset)
            nextgen_population.append(best_before)
        elif best_before_f == best_after_f:
            if best_before.size() < best_after.size():
                remove_selection(nextgen_population, dataset)
                nextgen_population.append(best_before)
        population = nextgen_population
        fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]
        if max(fitnesses) > best_of_run_f:
            best_of_run_f = max(fitnesses)
            best_of_run_gen = gen
            best_of_run = deepcopy(population[fitnesses.index(max(fitnesses))])

        print("\n\n")
        print("gen:", gen, ", best_of_run_f:", round(max(fitnesses), 3), ", size:", best_of_run.size())
        print("________________________")

        if best_of_run_f >= 0.7 and best_of_run_f <= 1.0:
            print("Critério de parada! ")
            break

    print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run attained at gen " + str(
        best_of_run_gen) + \
          " has f=" + str(round(best_of_run_f, 3)) + \
          " and has size = " + str(best_of_run.size())
          )
    best_of_run.print_tree()
    best_of_run.write_txt(goal, best_of_run_gen, best_of_run_f)
    return best_of_run_f, best_of_run_gen, best_of_run

"""
def mean(x):
    if type(x) == int:
        return x
    else:
        if len(x):
            return sum(x) / len(x)
        else:
            return 0

def std_dev(x):
    if type(x) == int:
        return 0
    else:
        math.sqrt(
            sum(
                list(
                    map(
                        lambda y: (y - mean(x)) ** 2, x
                    )
                )
            ) / len(x)
        )
"""

if __name__ == '__main__':

    #terminal_set = ['x', 'y', -1, 0, 1]
    #terminal_set = ['x', -5, -2, -1, 0, 1, 2, 5]
    terminal_set = ['x', 'y', -1, 0,  1]
    function_set = [plus, minus, times, division]
    N = 31
    gens = []
    plos = []
    best_of_runs = []
    sizes = []

    for _ in range(N):
        plo, gen, best_of_run = main()
        gens.append(gen)
        plos.append(plo)
        sizes.append(best_of_run.size())
        best_of_runs.append(best_of_run)
    mean_plo = mean(plos)
    std_dev_plo = stdev(plos)

    mean_gen = mean(gens)
    std_dev_gen = stdev(gens)

    mean_size = mean(sizes)
    std_dev_size = stdev(sizes)

    with open("info.txt", "w+") as file:
        file.write(f"Fitness (média +- std_dev): {mean_plo} " + u"\u00B1" + f" {std_dev_plo}")
        file.write(f"\nGerações (média +- std_dev): {mean_gen} " + u"\u00B1" + f" {std_dev_gen}")
        file.write(f"\nTamanho (média +- std_dev): {mean_size} " + u"\u00B1" + f" {std_dev_size}")

    print(f"Média +- std_dev best_fitness: {mean_plo} " + u"\u00B1" + f" {std_dev_plo}")
    print(f"Média +- std_dev gen: {mean_gen} " + u"\u00B1" + f" {std_dev_gen}")

    fig = plt.figure()
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212, sharex=ax1)

    ax1.plot(gens)
    plt.xlabel("Número de execuções")
    ax1.set_ylabel("Gerações")
    ax2.plot([best_of_runs[i].size() for i in range(len(best_of_runs))])
    ax2.set_ylabel("Tamanho da árvore")
    ax1.get_shared_x_axes().join(ax1, ax2)
    ax1.set_xticklabels([])
    plt.show()



