from utils.utils import create_train_transcriptions_df, create_train_correspondances_df, create_training_labels_df, create_test_transcriptions_df, create_test_correspondances_df


if __name__ == "__main__":
    # start by creating dataframes from the json files
    print("Creating dataframes from json files...")
    create_train_transcriptions_df()
    create_train_correspondances_df()
    create_training_labels_df()
    create_test_transcriptions_df()
    create_test_correspondances_df()