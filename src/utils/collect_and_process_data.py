import re
import locale
import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings("ignore")


def remove_cur(x):
    return str(x).replace("$", "")


# Declaration and lambda for dataframe formatting
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
number_format = lambda num: locale.atof(remove_cur(num))


"""
@param
input_file : Raw file having details of the listings

@output : Pre-processed dataframe, ready to build the model

Preprocessing of the data:
    We won't be requiring all the columns from the raw that influence the price.
    Some of the major features like, position, no of rooms, amenities, no. of ppl,
    cancellation policies etc define the price of the room. We shall consider only
    them and continue.

required_columns.txt: Expects it to be under data folder
    An external file which maintains the name of the required columns. It can be altered
    anytime based on the requirements

format:
    Just a lambda function to remove trailing new line characters or whitespaces
"""


def collect_and_process(input_file):
    # Read listings_summary file
    listings_summary_df = pd.read_csv(input_file)

    # required_columns = []
    with open("../../data/required_columns.txt") as infile:
        required_columns = map(lambda line: line.rstrip(), infile.readlines())

    # Setting the index to column id, Hence column id from the required columns is mandatory.
    indexed_summary_df = listings_summary_df[required_columns].set_index("id")

    """
        Format the price values from different price columns and normalize it.
        We can set some maximum price and also remove invalid prices like zeroes or negatives.
        This will distribute prices normally.
    """

    indexed_summary_df.security_deposit.fillna('$0.00', inplace=True)
    indexed_summary_df.cleaning_fee.fillna('$0.00', inplace=True)

    # Formatting numbers to float
    indexed_summary_df.price = indexed_summary_df.price.apply(number_format)
    indexed_summary_df.cleaning_fee = indexed_summary_df.cleaning_fee.apply(number_format)
    indexed_summary_df.security_deposit = indexed_summary_df.security_deposit.apply(number_format)
    indexed_summary_df.extra_people = indexed_summary_df.extra_people.apply(number_format)

    # Filter/drop out invalid or out of range prices
    price_centered_df = indexed_summary_df.drop(indexed_summary_df[(indexed_summary_df.price <= 0.0) |
                                                                   (indexed_summary_df.price > 500.0)].index)

    # If any of the values have null, drop those rows, otherwise, it won't be a proper calculation
    processed_df = price_centered_df.dropna(subset=['room_type', 'bathrooms', 'bedrooms', 'bed_type'])

    return processed_df
