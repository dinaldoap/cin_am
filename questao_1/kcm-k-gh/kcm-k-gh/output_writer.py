#!/usr/bin/env python
import numpy as np
import csv
from itertools import islice
import model as m
import os.path as p
from datetime import datetime as dt
import util as u

field_view = 'Table'
field_iteration = 'Iteration'
field_rai = 'ARI (Adjusted Rand Index)'
field_jkcm_k_gh = 'JKCM-K-GH'
field_g = "g{0}"
field_c_size = "c{0} (Size)"
field_1_s2 = "1/s2"
field_c = "c{0}"
field_c_original = "c{0} (Original)"
field_iterations_convergence = "# Iterations (Until convergence)"
field_start_time = "Start time"
field_finish_time = "Finish time"

fmt_datetime_log = "%d-%m-%Y %H:%M:%S.%f"
fmt_datetime_file_name = "%Y-%m-%d-%H-%M-%S"


class OutputWriter:
    """
    Representation of an Output Writer
    """

    def __init__(self, output_folder, data_normalized):
        datetime = dt.now().strftime(fmt_datetime_file_name)
        sufix = self.__get_normalization_sufix(data_normalized)
        file_name = "{0}{1}{2}.output".format(output_folder, datetime, sufix)
        self.__file = open(file_name, "a+", newline='')
        self.__has_header = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__file.close()

    def write_output(self, view, result, iteration, start_time, finish_time):
        nPartitions = len(result.g)
        fieldnames = self.__create_fieldnames(view, nPartitions)

        writer = csv.DictWriter(
            self.__file, fieldnames=fieldnames, delimiter=';', quoting=csv.QUOTE_ALL)

        if (not self.__has_header):
            writer.writeheader()
            self.__has_header = True

        row = dict()
        row[field_view] = view.name
        row[field_iteration] = iteration
        row[field_start_time] = start_time.strftime(fmt_datetime_log)
        row[field_finish_time] = finish_time.strftime(fmt_datetime_log)
        row[field_jkcm_k_gh] = result.jkcm_k_gh
        row[field_rai] = result.rai
        row[field_iterations_convergence] = result.iterations_convergence

        for i, g_i in enumerate(result.g):
            row[field_g.format(i + 1)] = g_i.data

        row[field_1_s2] = result._1_s2

        for i, c_i in enumerate(result.clusters):
            str_data = ' | '.join(str(x.data)
                                  for x in c_i.elements)
            str_orig_data = ' | '.join(str(x.original_data)
                                       for x in c_i.elements)
            row[field_c.format(i + 1)] = str_data
            row[field_c_original.format(i + 1)] = str_orig_data
            row[field_c_size.format(i + 1)] = c_i.size()

        writer.writerow(row)
        self.__file.flush()

    def __create_fieldnames(self, view, nPartitions):
        fieldnames = list()
        fieldnames.append(field_view)
        fieldnames.append(field_iteration)
        fieldnames.append(field_jkcm_k_gh)
        fieldnames.append(field_rai)
        fieldnames.append(field_iterations_convergence)
        fieldnames.append(field_start_time)
        fieldnames.append(field_finish_time)

        for i in range(nPartitions):
            fieldnames.append(field_g.format(i + 1))

        for i in range(nPartitions):
            fieldnames.append(field_c_size.format(i + 1))

        fieldnames.append(field_1_s2)

        for i in range(nPartitions):
            fieldnames.append(field_c.format(i + 1))

        for i in range(nPartitions):
            fieldnames.append(field_c_original.format(i + 1))

        return fieldnames

    def __get_normalization_sufix(self, normalized):
        if (normalized):
            sufix = "-normalized"
        else:
            sufix = "-not-normalized"
        return sufix
