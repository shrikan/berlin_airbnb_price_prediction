import collect_and_process_data
import data_morphing


if __name__ == '__main__':
    df = collect_and_process_data.collect_and_process("../../data/listings_summary.csv")
    df = data_morphing.add_distance(df)
    updated_df = data_morphing.get_area_from_description(df)

    data_morphing.update_amenities(updated_df)

    data_morphing.update_types(updated_df, 'property_type')
    data_morphing.update_types(updated_df, 'room_type')
    data_morphing.update_types(updated_df, 'bed_type')
    data_morphing.update_types(updated_df, 'cancellation_policy')

    data_morphing.update_bools(updated_df, 'instant_bookable')
    data_morphing.update_bools(updated_df, 'is_business_travel_ready')

