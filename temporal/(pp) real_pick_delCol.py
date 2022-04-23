import os
import csv
from tqdm import tqdm


main = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/'
# dataset = '5_real_fall21_x1/'
dataset = '1_proxy_rob537_x1/'
subfolder = 'pp1_split_x45cols/'

stage = 'GRASP/'

location = main + dataset + stage + subfolder

# First delete the columns
for filename in tqdm(sorted(os.listdir(location))):
    # print(filename)

    name = str(filename)
    var = name[-5]
    print(var)

    if var == "h":
        # These are WRENCH files, from which we'll delete columns 4 and 8
        print(filename)

        with open(location + filename, "r") as source:
            reader = csv.reader(source)

            new_name = name
            target_location = main + dataset + stage + '/pp1_split/'

            with open(target_location + new_name, "w", newline='') as result:
                writer = csv.writer(result)
                for r in reader:
                    # writer.writerow((r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]))
                    writer.writerow((r[0], r[1], r[2], r[3], r[5], r[6], r[7]))

    if var == "u":
        # These are IMU files, from which we'll delete columns 4
        print(filename)

        with open(location + filename, "r") as source:
            reader = csv.reader(source)

            new_name = name
            target_location = main + dataset + stage + '/pp1_split/'

            with open(target_location + new_name, "w", newline='') as result:
                writer = csv.writer(result)
                for r in reader:
                    # writer.writerow((r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]))
                    writer.writerow((r[0], r[1], r[2], r[3], r[5], r[6], r[7]))

    if var == "s":
        print(filename)

        with open(location + filename, "r") as source:
            reader = csv.reader(source)

            new_name = name
            target_location = main + dataset + stage + '/pp1_split/'

            with open(target_location + new_name, "w", newline='') as result:
                writer = csv.writer(result)
                for r in reader:
                    # writer.writerow((r[0], r[1], r[2], r[3]))
                    writer.writerow((r[0], r[1], r[2], r[3]))
