# @Time : 2/17/2022 10:04 PM
# @Author : Alejandro Velasquez



# Read the Real Picks metadata

# Data Location
import os
import csv

main = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/'
datasets = ['3_proxy_winter22_x1', '5_real_fall21_x1']
subfolder = '/metadata/'
# pic_file = 'apple_proxy_pick' + str(pick_number) + '_pick_' + str(topic) + '.csv'

location = main + datasets[1] + subfolder

for file in os.listdir(location):
    name = str(file)

    start = name.index('pick')
    end = name.index('meta')

    number_real = name[start+4:end-1]
    # print(number)

    # Open metadata and get the label
    rows=[]
    with open(location + file) as csv_file:
        # Create  a csv object
        csv_reader = csv.reader(csv_file, delimiter=',')
        # Extract each data row one by one
        for row in csv_reader:
            rows.append(row)

        if rows[1][10] == 's':
            result_real = 1
        else:
            result_real = 0

    # Now open the proxy files
    proxy_location = main + datasets[0] + subfolder
    for file_prox in os.listdir(proxy_location):

        name = str(file_prox)

        start = name.index('pick')
        end = name.index('-')

        number_proxy = name[start + 4:end]
        # print(number_proxy)
        # Only open those that have the same number
        if number_proxy == number_real:

            rows = []
            with open(proxy_location + file_prox) as csv_file:
                # Create  a csv object
                csv_reader = csv.reader(csv_file, delimiter=',')
                # Extract each data row one by one
                for row in csv_reader:
                    rows.append(row)

                # print(rows[1][10])

                if rows[1][10] == 's':
                    result_proxy = 1
                else:
                    result_proxy = 0

            if result_proxy == result_real:
                print("\nThere is a match:")
                print(result_proxy)
                print(file)
                print(file_prox)






# For each pick, read the result, and find the ones from the proxy that had the same result

