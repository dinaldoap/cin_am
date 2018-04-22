#!/usr/bin/env python
import numpy as np
import util


def kcm_k_gh(d, c, y, p):
    """
    Implementation of the variant with global vector of width hyper-parameters
    of the algorithm 'Gaussian kernel c-means with kernelization of
    the metric and automated computation of width parameters' (KCM-K-GH).

    Parameters
    ----------
    d : array_like
        The data set ``{ x1, ... xn }``.
    c :
        The number of clusters.
    y :
        A suitable parmeter ``γ``.
    p :
        Number of variables.
    """
    # INTIALIZATION:
    # Randomly select ``c`` distinct prototypes ``gi`` which belongs to D(1 <= i <= c):
    g = init_prototypes(d, c)

    # Initialize the width hyper-parameter vector ``s``:
    s = init_s(y, p)

    # Initialize clusters:
    clusters = calc_clusters(d, g, s)

    # ITERATION:
    test = None

    if __debug__:
        limit = 10

    while (test != 0):
        test = 0

        # Step 1: Representation
        g = calc_prototypes(clusters, g, s)

        # Step 2: Computation of the width hyper-parameters:
        s = calc_s(clusters, g, s, y, p)
        print(s)

        # Step 3: Allocation:
        new_clusters = calc_clusters(d, g, s)

        if (new_clusters != clusters):
            clusters = new_clusters
            test = 1

        if __debug__:
            #limit -= 1
            if limit <= 0:
                test = 0

    # TODO: Calcular numero de objetos no cluster e RAD:
    print("Cluster representatives `g`: %s" % g)
    print("Cluster objects number: %s" % '')
    print("Global vector of hyper-parameters `s`: %s" % s)
    print("Partitions `P`: %s" % clusters)
    print("Rand Adjusted Index (RAI): %s" % '')


def init_prototypes(d, c):
    """
    Randomly initalize prototypes based on ``d`` dataset
    and ``c`` number of clusters.

    Args:
        d: The dataset
        c: Number of elements to return
    """
    return util.take_random_elems(d, c)


def calc_prototypes(clusters, g, s):
    """
    Compute the cluster representatives ``g1 ... gn``.

    Formula:
        ``g_i = (sum_k(k_s(x_k, g_i) * x_k) / (sum_k(k_s(x_k, g_i))``

    Args:
        clusters: The clusters containing elements ``e``.
        g: The array of prototypes for clusters.
        s: The array of width hyper-parameters.

    Examples:
        >>> calc_prototypes([[[1.0, 2.0], [0.5, 2.1]], [[11.0, 22.0]], [[3.0, 5.0]]], [[0.8, 2.1], [4.0, 6.5], [9.3, 4.6]], [1.5, 2.8])
        [[0.7513091575342355, 2.0497381684931533], [11.0, 22.0], [3.0, 5.0]]
    """
    c = len(g)
    new_g = [[] for _ in range(c)]

    for i, g_i in enumerate(g):
        if g_i:
            e = clusters[i]
            new_g[i] = __calc_prototype_g_i(e, g_i, s)

    return new_g


def __calc_prototype_g_i(e, g_i, s):
    result = []

    if e:
        sum_1 = 0
        sum_2 = 0

        for e_k in e:
            ks_e_k = ks(e_k, g_i, s)
            sum_1 += (ks_e_k * np.array(e_k))
            sum_2 += ks_e_k

        temp = (sum_1 / sum_2)
        result = temp.tolist()

    return result


def init_s(y, p):
    """
    Initialize width hyper-parameter vector ``s``.

    Formula:
        Set ``(1/sj²) = (y)^(1/p)``, where (1 <= j <= p).
        ::: ``sj = sqrt(1/(y^(1/p)))``

    Args:
        y: Parameter ``γ``.
        p: Number of variables in samples.

    Returns:
        p-sized array containing ``s`` hyper-parameters for each variable.

    Examples:
        >>> init_s(1.5, 3)
        [0.9346552651840672, 0.9346552651840672, 0.9346552651840672]

        >>> init_s(0.83542, 5)
        [1.018144719524774, 1.018144719524774, 1.018144719524774, 1.018144719524774, 1.018144719524774]
    """
    temp = (y ** (1 / p))
    temp = (1 / temp)
    sj = np.sqrt(temp)
    return [sj] * p


def calc_s(clusters, g, s, y, p):
    """
    Computes the global vector of width hyper-parameters ``s``.

    Formula:
        ``(1 / s_j²) =
            (γ^(1 / p) * (prod_h( sum_i( sum_k( k_s(x_k, g_i) * (x_kh - g_ih)² ) ) ) ) ^ (1 / p))
            / (sum_i( sum_k( k_s(x_k, g_i) * (x_kh - g_ih)² ) ) ))``

    Args:
        clusters: The clusters containing the elements.
        g: The vector of prototypes representing the clusters.
        s: The actual vector of width hyper-parameters.
        y: The parameter ``γ``.

    Examples:
        >>> calc_s([[[1.0, 2.0], [0.5, 2.1]], [[3.0, 5.0]], [[11.0, 22.0]]], [[0.8, 2.1], [4.0, 6.5], [9.3, 4,6]], [1.5, 2.8], 0.04081632653061224, 2)
        [1.8920087087988386, 2.6161335544000908]
    """
    new_s = [[] for _ in s]

    for j in range(p):
        new_s[j] = __calc_s_j(j, clusters, g, s, y, p)

    return new_s


def __calc_s_j(j, clusters, g, s, y, p):
    # Calculate the number of parameters in prototype (which dimension is equals to elements):
    mult_h = 1

    for h in range(p):
        sum_h = __calc_s_j_sum_param(clusters, g, s, h)
        mult_h *= sum_h

    part_1 = (y ** (1/p)) * (mult_h ** (1/p))
    part_2 = __calc_s_j_sum_param(clusters, g, s, j)
    temp = 1 / (part_1 / part_2) # TODO: fazer tratamento de divisão por zero !!!
    s_j = np.sqrt(temp)
    return s_j


def __calc_s_j_sum_param(clusters, g, s, param):
    sum_i = 0

    for i, g_i in enumerate(g):
        sum_k = 0

        if g_i:
            e = clusters[i]

            for e_k in e:
                e_kh = e_k[param]
                g_ih = g_i[param]
                sum_k += (ks(e_k, g_i, s) * ((e_kh - g_ih) ** 2))

            sum_i += sum_k

    return sum_i


def calc_clusters(e, g, s):
    """
    Calculate clusters by verifying similarity with prototypes ``g``.

    Args:
        d: The dataset
        g: The vector containig the prototypes for clusters

    Examples:
        >>> calc_clusters([[1.0, 2.0], [3.0, 5.0], [11.0, 22.0], [0.5, 2.1]], [[0.8, 2.1], [4.0, 6.5], [9.3, 4.6]], [1.5, 2.8])
        [[[1.0, 2.0], [0.5, 2.1]], [[3.0, 5.0]], [[11.0, 22.0]]]
    """
    c = len(g)
    new_clusters = [[] for _ in range(c)]

    for e_k in e:
        # Find new cluster based on minimization of
        # the objective function (JKCM-G-GH):
        new_i = minimize_jkcm_g_gh(e_k, g, s)
        new_clusters[new_i].append(e_k)

    return new_clusters


def minimize_jkcm_g_gh(e_k, g, s):
    """
    The objective function ``JKCM-K-GH`` to minimize.

    Examples:
        >>> minimize_jkcm_g_gh([1.0, 2.0], [[0.8, 2.1], [4.0, 6.5], [9.3, 4,6]], [1.5, 2.8])
        0
    """
    c = len(g)
    jkcm_g_gh = np.array([None] * c)

    for i, g_i in enumerate(g):
        if g_i:
            # If `gi` is a valid prototype, calculate objective function:
            jkcm_g_gh[i] = __minimize_jkcm_g_gh_ek(e_k, g_i, s)
        else:
            # Otherwise, considers kernel as an infinity valye, in
            # order to avoid being considered by argmin()
            jkcm_g_gh[i] = util.infinity()

    # Gets the min value among calculated values
    return jkcm_g_gh.argmin()


def __minimize_jkcm_g_gh_ek(e_k, g_i, s):
    """
    The objective function ``JKCM-K-GH`` to minimize.
    """
    return 2 * (1 - ks(e_k, g_i, s))


def ks(x_l, x_k, s):
    """
    Kernel function, based on a Gaussian kernel function with a
    global vector of width hyper-parameters `s`.

    Formula:
        ``k_s(x_l, x_k) = exp(-1/2 * sum_j(1/sj² * (x_lj - x_kj)²))``

    Args:
        x_l:
        x_k:
        s: The global vector of width hyper-parameters.

    Examples:
        >>> ks([1.0, 2.0], [3.0, 4.0], [1.5, 2.8])
        0.31854519015889254
    """
    sum = 0

    for j, s_j in enumerate(s):
        x_lj = x_l[j]
        x_kj = x_k[j]
        sum += __ks_sum(x_lj, x_kj, s_j)

    result = (-1/2) * sum
    return np.exp(result)


def __ks_sum(x_lj, x_kj, s_j):
    s_j2 = (s_j ** 2)
    return ((1 / s_j2) * ((x_lj - x_kj) ** 2))


############# CÁLCULO DE γ: #############
def calc_y(p, d):
    """
    Calculate the parameter ``γ``.

    Args:
        p: Number of variables.
        d: The data set.

    Examples:
        >>> calc_y(2, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        0.04081632653061224
    """
    o2 = calc_o2(d)
    return (1 / o2) ** p


def calc_o2(d):
    """
    Calculate the ``σ²``.

    Average between the 0.1 and 0.9 quantile of ``||xl - xk||``,
    where ``l != k``.

    Args:
        d (array_like): The data set.

    Examples:
        >>> calc_o2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        4.949747468305833
    """
    # 1) Calcular distância Euclidiana de todos os pontos da amostra.
    #   Neste caso, será criada apenas a matriz da diagonal superior,
    #   uma vez que a diagonal inferior é espelhada da superior:
    distances = util.calculate_distance_matrix(d)

    # 2) Ordenar as distâncias encontradas
    distances_v = util.flatten_matrix(distances)
    distances_v = util.remove_none_and_zero_elems(distances_v)

    # 3) Calcular o 0,1 e 0,9 quantil
    distances_v.sort()
    perc_10 = np.percentile(distances_v, 10)
    perc_90 = np.percentile(distances_v, 90)

    # 4) Obter a média entre os quantis acima
    perc_avg = (perc_10 + perc_90) / 2

    return perc_avg


def read_file_data(file_name, start_row=0, start_col=0, end_col=None):
    """
    Read data from file.

    Examples:
        >>> read_file_data("test.data", 1, 1)
        [['1.0', '1.1'], ['2.0', '2.2'], ['3.0', '3.3']]

        >>> read_file_data("test.data", 1, 1, 1)
        [['1.0'], ['2.0'], ['3.0']]
    """
    output = list()

    with open(file_name, "r") as f:
        for i, line in enumerate(f):
            if (i >= start_row):
                _start_col = start_col
                _end_col = end_col

                # Split data by comma (,) delimiter:
                row_data = line.rstrip().split(",")

                # Check if start and end cols are valid,
                # and correct if needed:
                if (_start_col == None):
                    _start_col = 0
                if (_end_col == None):
                    _end_col = len(row_data) - 1

                # Truncate row data and add to output:
                row_data = row_data[_start_col:(_end_col+1)]
                output.append(row_data)

    return output


def run():

    # Data set:
    d = read_file_data("segmentation.data", 6, 1)
    d = util.parse_float(d)

    # Paramters:
    c = 7           # Number of clusters
    p = len(d[0])   # Number of parameters
    y = calc_y(p, d)

    # Algoritmo principal:
    kcm_k_gh(d, c, y, p)


run()

if __name__ == '__main__':
    import doctest
    doctest.testmod()
