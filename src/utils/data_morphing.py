from geopy.distance import great_circle
import importlib.util

scriptpath = "../model/regression_model.py"
spec = importlib.util.spec_from_file_location("get_missing_values", scriptpath)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def add_distance(df):
    df['distance'] = df.apply(lambda row: distance_to_mitte((row.latitude, row.longitude)), axis=1)

    # Once distance is added, no meaning for latitude and longitude, hence remove it
    df.drop(['latitude', 'longitude'], axis=1, inplace=True)

    return df


"""
    @param property_loc : A Tuple having latitude and longitude of the property
    
    :returns distance in kilometer from mitte
     
    Although is universally true, especially in Berlin, the price of the property
    is heavily dependant on the distance to mitte/Center. It is thus meaningful
    to morph distance to mitte. 
"""


def distance_to_mitte(property_loc):
    # Source: Wikipedia
    berlin_mitte_loc = (52.531677, 13.381777)

    return great_circle(berlin_mitte_loc, property_loc).km


"""
    @param df : Pre processed dataframe having description column in it
    
    :returns dataframe with new column area and removed column description
    
    The area of the rooms are usually mentioned in the description, it always
    thus makes more sense to extract the area from this field. The missing values
    can then be filled with regression method
"""


def get_area_from_description(df):
    area_pattern = "(\\d+\\s?[sS][qmM])"
    number_pattern = "[^\\d.]+"
    df['area'] = df['description'].str.extract(area_pattern, expand=True)
    df['area'] = df['area'].str.replace(number_pattern, "").astype(float)
    df = module.get_missing_values(df)
    df.drop(['description'], axis=1, inplace=True)
    return df


"""
    @param df : Preprocessed dataframe with amenities column having set of amenities
    
    :returns maps amenities values to number of amenities it has
    
    It is a straight forward approach to map this set of string values to an int value
    by taking number of amenities provided, as we know more the amenities, more will
    be the prices.
"""


def update_amenities(df):
    num_of_amenities = lambda amenity: float(len(amenity.strip("{}")
                                                 .replace('"', '').split(',')))

    df.amenities = df.amenities.apply(num_of_amenities)
    return df


"""
    @param df : Preprocessed dataframe with the column mentioned in 'col_name' being present
           col_name : it contains the column name being passed
           
    :returns updated dataframe with mapped values
    
    Here, value is replaced by number of occurrences of such values in the dataframe
"""


def update_types(df, col_name):
    types_dict = df[col_name].value_counts().to_dict()
    sum_of_types = lambda entry: types_dict[entry]

    df['modified_'+col_name] = df[col_name].apply(sum_of_types)
    df.drop([col_name], axis=1, inplace=True)
    df.rename(columns={'modified_' + col_name: col_name}, inplace=True)

    return df


"""
    @param df : Preprocessed dataframe with the column mentioned in 'col_name' being present
           col_name : it contains the column name being passed

    :returns updated dataframe with mapped values

    Here, bool values t and f are mapped to 1 and 0 respectively, to avoid None values, 
    even they are mapped to 0
"""


def update_bools(df, col_name):
    bool_dict = {None: 0, 'f': 0, 't': 1}
    bool_to_int = lambda entry: bool_dict[entry]

    df['modified_'+col_name] = df[col_name].apply(bool_to_int)
    df.drop([col_name], axis=1, inplace=True)
    df.rename(columns={'modified_' + col_name: col_name}, inplace=True)

    return df
