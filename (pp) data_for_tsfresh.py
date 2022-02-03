import os, glob
import csv
import pandas as pd

# Location of the data from the previous step
# Failed
# location = 'C:/Users/15416/PycharmProjects/PickApp/data/Real Apples Data/improved data/grasp/Data_with_33_cols/postprocess_3/failed/'
location = 'C:/Users/15416/PycharmProjects/PickApp/data/Apple Proxy Data/data_postprocess3 (only grasp_downsampled_ joined)/failed/'
target_location = 'C:/Users/15416/PycharmProjects/PickApp/data/Real Apples Data/improved data/grasp/Data_with_33_cols/postprocess_4_for_tsfresh/2 - failed_timeseries/'
label = 'false'
target_file = 'y_failed.csv'
i = 1

# #Success
# location = 'C:/Users/15416/PycharmProjects/PickApp/data/Real Apples Data/improved data/grasp/Data_with_33_cols/postprocess_3/successful/'
# location = 'C:/Users/15416/PycharmProjects/PickApp/data/Apple Proxy Data/data_postprocess3 (only grasp_downsampled_ joined)/successful/'
# target_location = 'C:/Users/15416/PycharmProjects/PickApp/data/Real Apples Data/improved data/grasp/Data_with_33_cols/postprocess_4_for_tsfresh/1 - success_timeseries/'
# label = 'true'
# target_file = 'y_success.csv'
# i = 194


# Step 1 Add columns
ids = []
results = []

# --- Step 1: Add ID, sort, and result column
for file in sorted(os.listdir(location)):

    # Get the number of the pick as an ID for the
    id = str(file)
    start = id.find('k')
    end = id.find('g')
    id = id[start+2:end-1]

    # print(id)

    with open(location + file, "r") as source:
        reader = csv.reader(source)
        next(reader, None)  # skip the headers

        with open(target_location + file, "w", newline='') as result:
            writer = csv.writer(result)
            sort = 0           # Replace the elapsed time with an integer, because ts-fresh requires needs the sort column like that
            for r in reader:
                writer.writerow((i, sort, r[2],  r[3],  r[4],  r[5],  r[6],  r[7],  r[8],  r[9],
                                 r[10], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18], r[19],
                                 r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27], r[28], r[29],
                                 r[30], r[31], r[32], r[33], label))
                sort +=1

    # List 1: indexes
    ids.append(i)
    results.append(label)
    i += 1

# df = pd.DataFrame({'id': ids, 'results': results})
# df.to_csv(target_location + target_file, index=False)

# print(df)

all_files = glob.glob(os.path.join(target_location, "*.csv"))

dataframe = pd.concat(map(pd.read_csv, all_files), ignore_index=True)

dataframe.to_csv(target_location + 'pleaaase.csv', index=False)