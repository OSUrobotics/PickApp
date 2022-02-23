# @Time : 2/17/2022 10:04 PM
# @Author : Alejandro Velasquez



# Read the Real Picks metadata

# Data Location
import os
import csv
import ast


def lowest_noise(location, pick):
    a = 1

    # First look at all the picks with the same number


    # Find the one with the lowest noise


    # Compare the outcome

def number_from_filename(filename):

    name = str(filename)

    start = name.index('pick')
    end = name.index('meta')

    number = name[start + 4:end - 1]

    return number

def pick_info_from_metadata(location, file, index):

    # Open metadata and get the label
    rows = []
    with open(location + file) as csv_file:

        # Create  a csv object
        csv_reader = csv.reader(csv_file, delimiter=',')

        # Extract each data row one by one
        for row in csv_reader:
            rows.append(row)

        info = rows[1][index]

    return info


if __name__ == "__main__":

    # --- Location of Data ----
    main = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/'
    datasets = ['3_proxy_winter22_x1', '5_real_fall21_x1']
    subfolder = '/metadata/'
    # pic_file = 'apple_proxy_pick' + str(pick_number) + '_pick_' + str(topic) + '.csv'

    location = main + datasets[1] + subfolder

    real_list = []
    proxy_list = []
    # ---- Same Outcome and pose ----
    for file in os.listdir(location):

        # Step 1: Get the real-pick number from the filename
        number_real = number_from_filename(file)

        # Step 2: Open metadata and get the label
        real_outcome = pick_info_from_metadata(location, file, 10)

        # Step 3: Now open the proxy files
        proxy_location = main + datasets[0] + subfolder
        cart_lowest_noise = 10000
        ang_lowest_noise = 10000
        proxy_id = '1'

        for file_prox in os.listdir(proxy_location):

            name = str(file_prox)
            start = name.index('pick')

            # First digit that relates to the real-picks
            end = name.index('-')
            number_proxy = name[start + 4:end]

            # Entire digit
            end = name.index('_m')
            number_proxy_noise = name[start + 4:end]

            # print(number_proxy_noise)
            # Only open those that have the same number

            if number_proxy == number_real:

                proxy_outcome = pick_info_from_metadata(proxy_location, file_prox, 10)
                proxy_noise = pick_info_from_metadata(proxy_location, file_prox, 16)

                # print(proxy_noise)
                # Measure the overall noise angular
                b = ast.literal_eval(proxy_noise)
                c = list(b)
                cart_noise = abs(c[0]) + abs(c[1]) + abs(c[2])
                ang_noise = abs(c[3]) + abs(c[4]) + abs(c[5])

                # if cart_noise < cart_lowest_noise and ang_noise < ang_lowest_noise and real_outcome == proxy_outcome:
                if real_outcome == proxy_outcome:
                    cart_lowest_noise = cart_noise
                    ang_lowest_noise = ang_noise
                    proxy_id = number_proxy_noise

                    print("\nThere is a match:")
                    print(proxy_outcome)
                    print(file)
                    print(file_prox)
                    print('Cart noise is:', cart_noise)
                    print('Ang noise is:', ang_noise)

                # NOTE: Unindent twice this if
                if not proxy_id == '1' and real_outcome == 'f':
                    proxy_list.append(proxy_id)
                    real_list.append(int(number_real))

    print(real_list)
    print(proxy_list)


    # ---- Check the picks that resulted in drops ----
    # location = main + datasets[0] + subfolder
    # for file in os.listdir(location):
    #
    #     # Step 2: Open metadata and get the label
    #     outcome = pick_info_from_metadata(location, file, 10)
    #     drop = pick_info_from_metadata(location, file, 8)
    #
    #     if outcome == 'f' and drop == 'y':
    #         print(file)