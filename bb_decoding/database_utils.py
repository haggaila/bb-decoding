# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
import csv
import pickle
import os.path
import pandas as pd
from typing import Dict, List, Optional, Any, Union

from pandas import DataFrame

S_DECODER_DATA_DB_FILENAME = "decoder.database.csv"
"""File name to use for the database of all decoder data files."""

S_DECODER_DATA_PREFIX = "decoder"
"""Prefix for the file names of all decoder data files."""

S_SIMULATION_DB_FILENAME = "simulation.database.csv"
"""File name to use for the database of all simulation result files."""

S_SIMULATION_PREFIX = "simulation"
"""Prefix for the file names of all simulation result files."""


def generate_decoding_paths(n_up_dirs=1):
    """Concatenate a data directory and figures directory path, and create the directories.

    Returns:
        A 3-tuple with the directory strings.
    """
    s_output_path = ""
    for _ in range(n_up_dirs):
        s_output_path += "../"
    s_output_path = (
        os.path.abspath(s_output_path + "output") + "/"
    )  # use the absolute path of the current file
    if not os.path.exists(s_output_path):
        os.mkdir(s_output_path)
    s_decoder_path = s_output_path + "decoders/"
    if not os.path.exists(s_decoder_path):
        os.mkdir(s_decoder_path)
    s_simulation_path = s_output_path + "simulations/"
    if not os.path.exists(s_simulation_path):
        os.mkdir(s_simulation_path)
    s_plot_path = s_output_path + "figures/"
    if not os.path.exists(s_plot_path):
        os.mkdir(s_plot_path)
    return (
        s_output_path,
        s_decoder_path,
        s_simulation_path,
        s_plot_path,
    )


def save_to_db(s_db_path: str, line_data: dict):
    """Save a data line into the .csv dataframe file using pandas.

    Args:
        s_db_path: The full filename for the .csv dataframe file.
        line_data: The dictionary giving one line of db data.
    """
    if not os.path.isfile(s_db_path):
        # New database, write header line based on metadata keys
        with open(s_db_path, "w") as f:
            header = line_data.keys()
            writer = csv.writer(f)
            writer.writerow(header)
            f.close()

    db_line = {}
    for key in line_data.keys():
        db_line[key] = [line_data[key]]
    line_df = pd.DataFrame(db_line)
    df = pd.read_csv(s_db_path)
    if df.empty:
        df = line_df
    else:
        df = pd.concat([df, line_df])
    df.to_csv(s_db_path, index=False)


def save_decoder_data(db_line: Dict, decoder_data: Dict):
    s_uuid = db_line["unique_id"]
    s_output_path, s_decoder_path, _, _ = generate_decoding_paths()

    s_data_filename = s_decoder_path + S_DECODER_DATA_PREFIX + "." + s_uuid + ".pkl"
    print("Saving data to ", s_data_filename)
    with open(s_data_filename, "wb") as fp:
        pickle.dump(decoder_data, fp)

    s_db_path = s_output_path + S_DECODER_DATA_DB_FILENAME
    save_to_db(s_db_path, db_line)


def load_decoder_data(s_uuid: str) -> Dict:
    _, s_decoder_path, _, _ = generate_decoding_paths()
    s_data_filename = s_decoder_path + S_DECODER_DATA_PREFIX + "." + s_uuid + ".pkl"
    print("Loading decoder data from ", s_data_filename)
    with open(s_data_filename, "rb") as fp:
        decoder_data = pickle.load(fp)
    return decoder_data


def load_simulation_data(s_uuid: str, n_up_dirs=2) -> Dict:
    _, _, s_simulation_path, _ = generate_decoding_paths(n_up_dirs=n_up_dirs)
    s_data_filename = s_simulation_path + S_SIMULATION_PREFIX + "." + s_uuid + ".pkl"
    print("Loading simulation data from ", s_data_filename)
    with open(s_data_filename, "rb") as fp:
        decoder_data = pickle.load(fp)
    return decoder_data


def save_simulation_data(db_line: Dict, simulation_results: Dict):
    s_uuid = db_line["unique_id"]
    s_output_path, _, s_simulation_path, _ = generate_decoding_paths()

    s_data_filename = s_simulation_path + S_SIMULATION_PREFIX + "." + s_uuid + ".pkl"
    print("Saving data to ", s_data_filename)
    with open(s_data_filename, "wb") as fp:
        pickle.dump(simulation_results, fp)

    s_db_path = s_output_path + S_SIMULATION_DB_FILENAME
    save_to_db(s_db_path, db_line)


def query_simulations_by_name(
    s_simulations_name: str,
    sort_by: Optional[Any] = None,
    ascending: Union[bool, List[bool]] = True,
    na_position="last",
    parse_dates=None,
):
    """Find the simulations according to the criteria query and metadata dictionaries in a list.

    Args:
        s_simulations_name: name field to query
        sort_by: If not None, defines sorting using the method sort_values() of the data frame.
        ascending: If sort_by is not None, defines an option for the sort
        na_position: If sort_by is not None, defines an option for the sort
        parse_dates: Date fields.

    Returns:
            A list with the relevant simulation dicts.
    """

    s_output_path, _, _, _ = generate_decoding_paths(n_up_dirs=1)
    s_db_path = s_output_path + S_SIMULATION_DB_FILENAME

    df = pd.read_csv(s_db_path, parse_dates=parse_dates)
    df_2 = df.loc[df["name"] == s_simulations_name]
    if sort_by is not None:
        df_3 = df_2.sort_values(sort_by, ascending=ascending, na_position=na_position)
    else:
        df_3 = df_2
    return df_3


def find_simulation_id(files: List[str], s_filter_query: str):
    """
        finds the specific simulations id according to the query defined and stores the id's in a list

        Args:
            files: A list with db (.csv) files.
            s_filter_query: A string with the desired query.

        Returns:
            selected_id: A list with the relevant simulation id's.
    """
    selected_id = []
    for file in files:
        df = pd.read_csv(file)
        df1 = df.query(s_filter_query)
        selected_id.extend(df1.id.unique())
    return selected_id
