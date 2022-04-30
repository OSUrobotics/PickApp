""""
Code to rename a bunch of files with a particular name
References: https://www.geeksforgeeks.org/rename-multiple-files-using-python/
"""

import os

# Function to rename multiple files
def main():

    location = '/home/avl/PycharmProjects/AppleProxy/bagfiles/'
    # old = 'apple_proxy'
    # new = 'real_apple'

    old = 'fall21_fall21_'
    new = 'fall21_'

    for count, filename in enumerate(os.listdir(location)):

        print("\nOriginal file %i is %s" %(count, filename))
        new_name = filename.replace(old, new)
        print("The new name is %s" % new_name)

        # Swap 'old' with 'new'
        src = location + filename
        target = location + new_name
        os.rename(src, target)


if __name__ == '__main__':
    main()
