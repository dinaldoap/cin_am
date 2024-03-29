#!/usr/bin/env python
import numpy as np
import util as u
import csv
from itertools import islice
import model as m
from sklearn.metrics import cluster as cluster
from sys import float_info as finfo
import output_writer as ow
from datetime import datetime as dt
import argparse

data_cache = dict()
y_cache = dict()


def kcm_k_gh(data_file, view, partitions, normalize_data):
    # Data set:
    d = provide_data(data_file, view, normalize_data)

    # Paramters:
    print("Setting up...")
    c = partitions              # Number of clusters
    p = len(d[0].data)          # Number of parameters
    y = provide_y(view, p, d)   # Parameter 'y'

    # Algoritmo principal:
    print("Running KCM-K-GH...")
    result = __kcm_k_gh(d, c, y, p)

    return result


def normalize_data(d):
    cols_totals = sum_cols(d)
    samples = list()
    sample = None

    for _, e_k in enumerate(d):
        sample = m.Sample(e_k.label, e_k.new_label, e_k.data)

        for i, x_i in enumerate(e_k.data):
            new_val = x_i / cols_totals[i]

            # if (new_val == 0):
            #     new_val = finfo.min

            sample.data[i] = new_val

        samples.append(sample)

    return samples


def sum_cols(d):
    p = len(d[0].data)
    totals = [0 for _ in range(p)]

    for _, e_k in enumerate(d):
        for i, x_i in enumerate(e_k.data):
            totals[i] += x_i

    return totals


def __kcm_k_gh(d, c, y, p):
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
    (g, _1_s2, d) = init_kcm_k_gh(d, c, y, p)

    # ITERATION:
    iteration = 1
    test = None

    while (test != 0):
        print(" > Iteration %s..." % iteration)
        test = 0

        # Step 1: Representation
        g = calc_prototypes(d, g, _1_s2)

        # Step 2: Computation of width hyper-parameters:
        _1_s2 = calc_1_s2(d, g, _1_s2, y, p)

        # Step 3: Allocation:
        (d, test) = calc_clusters(d, g, _1_s2)

        iteration += 1

    return process_results(d, g, _1_s2, iteration)


def init_kcm_k_gh(d, c, y, p):
    """
    Process initialization for KCM-K-GH algorithm.

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
    # Randomly select ``c`` distinct prototypes ``gi`` 
    # which belongs to D(1 <= i <= c):
    g = init_prototypes(d, c)

    # Initialize the width hyper-parameter vector ``s``:
    _1_s2 = init_1_s2(y, p)

    # Initialize clusters:
    (d, _) = calc_clusters(d, g, _1_s2)

    return (g, _1_s2, d)


def process_results(d, g, _1_s2, numIterations):
    jkcm_k_gh = calc_jkcm_k_gh(d, g, _1_s2)
    rai = calc_rai(d)
    clusters = create_clusters(d, g)
    return m.ExecutionResult(jkcm_k_gh, rai, clusters, g, _1_s2, numIterations)


def print_output(result):
    # Cluster representatives:
    __print_simple_output("Cluster representatives (`g`)", result.g, 'data')

    # Cluster objects number:
    print_clusters_objs_num(result.clusters)

    # Hyper parameters vector:
    print("Global vector of hyper-parameters (`1/s²`): \n%s\n" % result._1_s2)

    # Rand Adjusted Index (RAI):
    print("Rand Adjusted Index (RAI): %s\n" % result.rai)

    # Partitions:
    print_partitios(result.clusters)


def print_clusters_objs_num(clusters):
    output = ""

    for i, cluster in enumerate(clusters):
        output = output + "  {} -> {} element(s)\n".format(i, cluster.size())

    print("Cluster objects number: \n%s" % output)


def print_partitios(clusters):
    output = ""

    for i, cluster in enumerate(clusters):
        output += "  {} -> \n".format(i)

        for e_k in cluster.elements:
            output += "    {}\n".format(e_k.data)

    print("Partitios (`P`): \n%s" % output)


def __print_simple_output(msg, params, attr=None):
    output = ""

    for i, p in enumerate(params):
        if (attr):
            val = getattr(p, attr)
        else:
            val = p

        output = output + "  {} -> {} \n".format(i, val)

    print("%s: \n%s" % (msg, output))


def init_prototypes(d, c):
    """
    Randomly initalize prototypes based on ``d`` dataset
    and ``c`` number of clusters.

    Args:
        d: The dataset
        c: Number of elements to return
    """
    samples = u.take_random_elems(d, c)
    g = list()

    for i, sample in enumerate(samples):
        g_i = m.Prototype(i, sample.data)
        g.append(g_i)

    return g


def calc_prototypes(d, g, _1_s2):
    """
    Compute the cluster representatives ``g1 ... gn``.

    Formula:
        ``g_i = (sum_k(k_s(x_k, g_i) * x_k) / (sum_k(k_s(x_k, g_i))``

    Args:
        clusters: The clusters containing elements ``e``.
        g: The array of prototypes for clusters.
        s: The array of width hyper-parameters.

    Examples:
        >>> calc_prototypes([[[1.0, 2.0], [0.5, 2.1]], [[11.0, 22.0]], [[3.0, 5.0]]], [[0.8, 2.1], [4.0, 6.5], [9.3, 4.6]], [0.044, 0.032])
        [[0.7501174999913481, 2.0499765000017303], [11.0, 22.0], [3.0, 5.0]]
    """
    c = len(g)
    acc_1 = [0 for _ in range(c)]
    acc_2 = [0 for _ in range(c)]

    for e_k in d:
        i = e_k.new_label
        acc_1[i], acc_2[i] = __calc_prototype_g_i(
            e_k, g[i], _1_s2, acc_1[i], acc_2[i])

    new_g = [None for _ in range(c)]

    # TODO: division by zero
    for i, _ in enumerate(new_g):
        temp = (acc_1[i] / acc_2[i])
        data = temp.tolist()
        new_g[i] = m.Prototype(i, data)

    return new_g


def __calc_prototype_g_i(e_k, g_i, _1_s2, acc_1, acc_2):
    g_i_data = g_i.data
    e_k_data = e_k.data

    ks_e_k = ks(e_k_data, g_i_data, _1_s2)
    new_acc_1 = acc_1 + (ks_e_k * np.array(e_k_data))
    new_acc_2 = acc_2 + ks_e_k
    return (new_acc_1, new_acc_2)


# def __calc_prototype_g_i(e, g_i, _1_s2):
#     result = []

#     if e:
#         sum_1 = 0
#         sum_2 = 0

#         for e_k in e:
#             e_k_data = e_k.data
#             ks_e_k = ks(e_k_data, g_i, _1_s2)
#             sum_1 += (ks_e_k * np.array(e_k_data))
#             sum_2 += ks_e_k

#         temp = (sum_1 / sum_2)
#         result = temp.tolist()

#     return result


def init_1_s2(y, p):
    """
    Initialize width hyper-parameter vector ``s``.

    Formula:
        Set ``(1/sj²) = (y)^(1/p)``, where (1 <= j <= p).

    Args:
        y: Parameter ``γ``.
        p: Number of variables in samples.

    Returns:
        p-sized array containing ``s`` hyper-parameters for each variable.

    Examples:
        >>> init_1_s2(1.5, 3)
        [1.1447142425533319, 1.1447142425533319, 1.1447142425533319]

        >>> init_1_s2(0.83542, 5)
        [0.9646748886937874, 0.9646748886937874, 0.9646748886937874,
            0.9646748886937874, 0.9646748886937874]
    """
    _1_s2_j = (y ** (1 / p))
    return [_1_s2_j] * p


def calc_1_s2(d, g, _1_s2, y, p):
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
        >>> calc_1_s2([[[1.0, 2.0], [0.5, 2.1]], [[3.0, 5.0]], [[11.0, 22.0]]], [[0.8, 2.1], [4.0, 6.5], [9.3, 4,6]], [0.044, 0.032], 0.04081632653061224, 2)
        [0.3793162998795441, 0.10760498967108424]
    """
    new_s = [[] for _ in _1_s2]

    part_1 = __calc_1_s2_j_part_1(d, g, _1_s2, y, p)

    for j in range(p):
        part_2 = __calc_1_s2_j_part_2(d, g, _1_s2, j)
        new_s[j] = (part_1 / part_2)

    return new_s


def __calc_1_s2_j_part_1(d, g, _1_s2, y, p):
    c = len(g)
    acc_1 = [[0] * c for _ in range(p)]

    # Sum elements according to parameter (h) and cluster (i),
    # accumulating for post-processing:
    for e_k in d:
        i = e_k.new_label

        for h in range(p):
            acc_1[h][i] = __calc_s_j_sum_param(
                e_k, g[i], _1_s2, h, acc_1[h][i])

    # Post-process accumulated values, calculating the product
    # among summed parameters (h):
    mult_h = 1

    for h in range(p):
        sum_h = sum(acc_1[h])
        mult_h *= sum_h

    # Then apply formula for `part 1`:
    part_1 = (y ** (1/p)) * (mult_h ** (1/p))

    return part_1


def __calc_1_s2_j_part_2(d, g, _1_s2, j):
    c = len(g)
    acc_2 = [0 for _ in range(c)]

    for e_k in d:
        i = e_k.new_label
        acc_2[i] = __calc_s_j_sum_param(e_k, g[i], _1_s2, j, acc_2[i])

    return sum(acc_2)


def __calc_s_j_sum_param(e_k, g_i, _1_s2, param, acc):
    e_k_data = e_k.data
    g_i_data = g_i.data

    e_kh = e_k_data[param]
    g_ih = g_i_data[param]
    new_acc = acc + (ks(e_k_data, g_i_data, _1_s2) * ((e_kh - g_ih) ** 2))

    return new_acc


def calc_clusters(d, g, _1_s2):
    """
    Calculate clusters by verifying similarity with prototypes ``g``.

    Args:
        d: The dataset
        g: The vector containig the prototypes for clusters

    Examples:
        >>> calc_clusters([[1.0, 2.0], [3.0, 5.0], [11.0, 22.0], [0.5, 2.1]], [[0.8, 2.1], [4.0, 6.5], [9.3, 4.6]], [0.044, 0.032])
        [[[1.0, 2.0], [0.5, 2.1]], [[3.0, 5.0]], [[11.0, 22.0]]]
    """
    has_changes = 0

    for e_k in d:
        # Find new cluster based on minimization of
        # the objective function (JKCM-G-GH):
        new_i = minimize_jkcm_k_gh_for_ek(e_k, g, _1_s2)

        if (new_i != e_k.new_label):
            e_k.new_label = new_i
            has_changes = 1

    return (d, has_changes)


def calc_jkcm_k_gh(d, g, _1_s2):
    """
    The objective function ``JKCM-K-GH``.
    """
    jkcm_k_gh_clusters = calc_jkcm_k_gh_for_clusters(d, g, _1_s2)
    return jkcm_k_gh_clusters.sum()


def calc_jkcm_k_gh_for_clusters(d, g, _1_s2):
    """
    The objective function ``JKCM-K-GH``.
    """
    c = len(g)
    acc = np.array([0] * c, np.float)

    for e_k in d:
        i = e_k.new_label
        g_i = g[i]
        jkcm_k_gh = __calc_jkcm_k_gh_ek(e_k, g_i, _1_s2)
        acc[i] += jkcm_k_gh

    return acc


def minimize_jkcm_k_gh_for_ek(e_k, g, _1_s2):
    """
    The objective function ``JKCM-K-GH`` calculated for all clusters.
    """
    c = len(g)
    jkcm_k_gh_clusters = np.array([None] * c)

    for i, g_i in enumerate(g):
        jkcm_k_gh_clusters[i] = __calc_jkcm_k_gh_ek(e_k, g_i, _1_s2)

    # Gets the min value among calculated values:
    return jkcm_k_gh_clusters.argmin()


def __calc_jkcm_k_gh_ek(e_k, g_i, _1_s2):
    """
    The objective function ``JKCM-K-GH`` to minimize.
    """
    e_k_data = e_k.data
    g_i_data = g_i.data
    return 2 * (1 - ks(e_k_data, g_i_data, _1_s2))


def ks(x_l, x_k, _1_s2):
    """
    Kernel function, based on a Gaussian kernel function with a
    global vector of width hyper-parameters `s`.

    Formula:
        ``k_s(x_l, x_k) = exp(-1/2 * sum_j(1/sj² * (x_lj - x_kj)²))``

    Args:
        x_l:
        x_k:
        _1_s2: The global vector of width hyper-parameters.

    Examples:
        >>> ks([1.0, 2.0], [3.0, 4.0], [0.044, 0.032])
        0.8589882807411234
    """
    sum = 0

    for j, _1_s2_j in enumerate(_1_s2):
        x_lj = x_l[j]
        x_kj = x_k[j]
        sum += __ks_sum(x_lj, x_kj, _1_s2_j)

    result = (-1/2) * sum
    return np.exp(result)


def __ks_sum(x_lj, x_kj, _1_s2_j):
    return (_1_s2_j * ((x_lj - x_kj) ** 2))


def create_clusters(d, g):
    c = len(g)
    clusters = [m.Cluster() for _ in range(c)]

    for e_k in d:
        i = e_k.new_label
        clusters[i].append(e_k)

    return clusters


def calc_rai(d):
    labels_true = [e_k.label for e_k in d]
    labels_pred = [e_k.new_label for e_k in d]
    rai = cluster.adjusted_rand_score(labels_true, labels_pred)
    return rai

############# CÁLCULO DE γ: #############


def provide_y(view, p, d):
    if (view.name not in y_cache):
        print("Calculating `y`...")
        y_cache[view.name] = calc_y(p, d)

    return y_cache[view.name]


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
    """
    # 1) Calcular distância Euclidiana entre todos os pontos da
    # amostra, dois a dois:
    distances_v = u.calculate_distance_matrix(d)
    #distances_v = u.remove_none_and_zero_elems(distances_v)

    # 2) Ordenar as distâncias encontradas:
    distances_v.sort()

    # 3) Calcular o 0,1 e 0,9 quantil:
    perc_10 = np.percentile(distances_v, 10)
    perc_90 = np.percentile(distances_v, 90)

    # 4) Obter a média entre os quantis acima:
    perc_avg = (perc_10 + perc_90) / 2

    return perc_avg


def provide_data(file_name, view, normalize):
    if (view.name not in data_cache):
        # Loading step:
        print("Loading data for '%s'..." % view.name)
        data = read_file_data(file_name, view)

        # Normalization step:
        if (normalize):
            print("Normalizing data...")
            data = normalize_data(data)

        data_cache[view.name] = data

    return data_cache[view.name]


def read_file_data(file_name, view):
    """
    Read data from file.

    Examples:
        >>> read_file_data("test.data", 1, 1)
        [['1.0', '1.1'], ['2.0', '2.2'], ['3.0', '3.3']]

        >>> read_file_data("test.data", 1, 1, 1)
        [['1.0'], ['2.0'], ['3.0']]
    """
    output = list()
    known_labels = dict()
    remove_cols = None

    with open(file_name, "r") as f:
        reader = csv.reader(f)

        for row in islice(reader, view.startRow, None):
            # Check if start and end cols are valid,
            # and correct if needed:
            if (remove_cols == None):
                _all_cols = range(0, len(row))
                remove_cols = [c for c in _all_cols if c not in view.cols()]

            # Truncate row data and add to output:
            data = u.remove_cols(row, remove_cols)
            data = u.parse_float(data)

            # Get label from row:
            label_name = row[view.labelCol]

            # Transcript row to an index-based representation:
            label = transcript_label(known_labels, label_name)

            # Create new sample instance, adding it to return:
            sample = m.Sample(label, None, data)
            output.append(sample)

    return output


def transcript_label(labels, label_name):
    if label_name in labels:
        label = labels[label_name]
    else:
        label = len(labels)
        labels[label_name] = label

    return label


def print_step(view, step):
    print("")
    print("------------------------------------------------")
    print(" %s - Iteration %s:" % (view.name, step))
    print("------------------------------------------------")


def run(data_file, output_folder, nIterations, nPartitions, normalize_data):
    completeView = m.View("Complete View", u.range_list(1, 21), 0, 5, [3, 4, 5])
    shapeView = m.View("Shape View", u.range_list(1, 10), 0, 5, [3, 4, 5])
    rgbView = m.View("RGB View", u.range_list(11, 21), 0, 5)
    views = [completeView, shapeView, rgbView]

    print("Starting processing...")

    with ow.OutputWriter(output_folder, normalize_data) as writer:
        for view in views:
            for i in range(nIterations):
                success = False
                iteration = i + 1

                print_step(view, iteration)

                while not success:
                    try:
                        start_time = dt.now()
                        result = kcm_k_gh(
                            data_file, view, nPartitions, normalize_data)
                        finish_time = dt.now()
                        success = True
                    except ZeroDivisionError as e:
                        print("Erro: %s" % (e))

                # Write output to file:
                writer.write_output(view, result, iteration,
                                    start_time, finish_time)

    print("\nProcessing completed.")


def watch_args():
    parser = argparse.ArgumentParser(
        description='Executes KCM-K-GH algorithm.')
    parser.add_argument('data_file', metavar='-d', type=str,
                        help='Path to the data to be processed.')
    parser.add_argument('output_folder', metavar='-o', type=str,
                        help='Path to the folder were results must be written.')
    parser.add_argument('nIterations', metavar='-i', type=int,
                        help='Number of iterations to process.')
    parser.add_argument('nPartitions', metavar='-c', type=int,
                        help='Number of clusters to create.')
    parser.add_argument('normalize', metavar='-n', type=str, default='n', choices=['s', 'n'],
                        help='Normalize data (s/n).')

    args = parser.parse_args()

    run(args.data_file, args.output_folder, args.nIterations,
        args.nPartitions, (args.normalize == 's'))


watch_args()
#run('data\\segmentation.test', 15, 7, True)
