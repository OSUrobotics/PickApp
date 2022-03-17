# @Author : Alejandro Velasquez

"""
1 - Plots the characteristic angles of the picks w.r.t. base
2 - Plots the characteristic angles of the picks w.r.t. hand
3 - Plots a scatterplot of all the characteristic angles and shows the k-means.
"""

# Some references
# https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
# https://stackoverflow.com/questions/32424670/python-matplotlib-drawing-3d-sphere-with-circumferences
# https://stackoverflow.com/questions/54970401/matplotlib-scatter-plot-with-xyz-axis-lines-through-origin-0-0-0-and-axis-proj

# Math related Packages
import numpy as np
from sklearn.cluster import KMeans
# File handling related packages
import csv
# Plot Packages
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits import mplot3d

def exam(x):
    return x**2


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def draw_in_hand(i, plot=True):
    """
    Draws all the pick vectors: Normal to hand, stem and gravity within the
    hand's cframe, however the gravity doesn't point always down
    :param i:
    :return:
    """
    apple_center = i[0]
    apple_calix = i[1]
    apple_stem = i[2]
    origin = i[3]
    gravity = i[4]

    # Get rid of characters
    apple_center = apple_center.strip('][').split(',')
    apple_calix = apple_calix.strip('][').split(',')
    apple_stem = apple_stem.strip('][').split(',')
    origin = origin.strip('][').split(',')
    gravity = gravity.strip('][').split(',')

    # ---- Step 1: Draw the Stem Vector ----
    calix = np.array([float(apple_calix[0]), float(apple_calix[1]), float(apple_calix[2])])
    stem = np.array([float(apple_stem[0]), float(apple_stem[1]), float(apple_stem[2])])
    stem_vector = np.subtract(stem, calix)
    stem_vector = stem_vector / np.linalg.norm(stem_vector)  # Normalize its magnitude

    # ---- Step 2: Draw the gravity Vector ----
    point_A = np.array([float(origin[0]), float(origin[1]), float(origin[2])])
    point_B = np.array([float(gravity[0]), float(gravity[1]), float(gravity[2])])
    gravity_vector = np.subtract(point_B, point_A)
    gravity_vector = gravity_vector / np.linalg.norm(gravity_vector)  # Normalize its magnitude

    # ---- Step 3: Draw the apple -----
    a = float(apple_center[0])
    b = float(apple_center[1])
    c = float(apple_center[2])
    a = b = c = 0

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    diam = 0.5
    x_a = (diam * np.outer(np.cos(u), np.sin(v))) + a
    y_a = (diam * np.outer(np.sin(u), np.sin(v))) + b
    z_a = (diam * np.outer(np.ones(np.size(u)), np.cos(v))) + c

    # ---- Step 5: Draw the Axes lines, which represent the Hand's coordinate frame ----
    x, y, z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    u, v, w = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 3]])

    # ---- Step 6: Get some Math
    hand_vector = np.array([0, 0, 1])

    # Angle between Hand and Stem
    dot_product = np.dot(hand_vector, stem_vector)
    handToStem_angle = np.arccos(dot_product)
    handToStem_angle = np.degrees(handToStem_angle)
    # print('The angle between the Hand and Stem is %.0f\N{DEGREE SIGN}' % handToStem_angle)

    # Angle between Hand and Gravity vector
    dot_product = np.dot(hand_vector, gravity_vector)
    handToGravity_angle = np.arccos(dot_product)
    handToGravity_angle = np.degrees(handToGravity_angle)
    # print('The angle between the Hand and Gravity is %.0f\N{DEGREE SIGN}' % handToGravity_angle)

    # Angle between Stem and Gravity vector
    dot_product = np.dot(stem_vector, gravity_vector)
    stemToGravity_angle = np.arccos(dot_product)
    stemToGravity_angle = np.degrees(stemToGravity_angle)
    # print('The angle between the Stem and Gravity is %.0f\N{DEGREE SIGN}' % stemToGravity_angle)

    if plot:
        # ---- Step 0: Initialize Figure
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel('Hand X ')
        ax.set_ylabel('Hand Y ')
        ax.set_zlabel('Hand Z ')
        # Get rid of the ticks
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.axes.zaxis.set_ticklabels([])

        # Draw the Stem Vector
        ax.quiver(0, 0, 0, stem_vector[0], stem_vector[1], stem_vector[2], length=1, color='b')
        ax.text(stem_vector[0], stem_vector[1], stem_vector[2], "Stem", color='b', size=15, zorder=1)

        # Draw Gravity Vector
        ax.quiver(0, 0, 0, gravity_vector[0], gravity_vector[1], gravity_vector[2], length=1, color='r')
        ax.text(gravity_vector[0], gravity_vector[1], gravity_vector[2], "Gravity", color='r', size=15, zorder=1)

        # Draw the Apple and its center
        ax.plot_surface(x_a, y_a, z_a, rstride=4, cstride=4, color='r', linewidth=0, alpha=0.2)
        ax.scatter(a, b, c, color="g", s=100)

        # Draw the Axes Lines
        ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.1, color="black")
        ax.text(0, 0, 1.5, "Normal to Hand", color='k', size=15, zorder=1)

        # Titles
        plt.suptitle("Stem and gravity w.r.t hand - Pick %i" % (k + 1))
        plt.title(
            'Hand-Stem: %.0f\N{DEGREE SIGN} , Hand-Gravity: %.0f\N{DEGREE SIGN}, Stem-Gravity %0.f\N{DEGREE SIGN}' % (
                handToStem_angle, handToGravity_angle, stemToGravity_angle))

    return handToStem_angle, handToGravity_angle, stemToGravity_angle


def draw_in_base(i, handToStem_angle, handToGravity_angle, stemToGravity_angle):
    """
    Draws all the pick vectors: Normal to hand, stem and gravity within the
    baselink's cframe, so the gravity is always pointing down
    :param i:
    :return:
    """
    apple_center = i[0]
    apple_calix = i[1]
    apple_stem = i[2]
    hand_origin = i[3]
    hand_x = i[4]
    hand_y = i[5]
    hand_z = i[6]

    # Get rid of characters
    apple_center = apple_center.strip("][").split(",")
    apple_calix = apple_calix.strip("']['").split("', '")
    apple_stem = apple_stem.strip("']['").split("', '")
    hand_origin = hand_origin.strip('][').split(',')
    hand_x = hand_x.strip('][').split(',')
    hand_y = hand_y.strip('][').split(',')
    hand_z = hand_z.strip('][').split(',')

    # ---- Step 0: Initialize Figure
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('Base X ')
    ax.set_ylabel('Base Y ')
    ax.set_zlabel('Base Z ')
    # Get rid of the ticks
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])

    # ---- Step 1: Draw the gravity Vector ----
    gravity_vector = np.array([0, 0, -1])
    ax.quiver(0, 0, 0, gravity_vector[0], gravity_vector[1], gravity_vector[2], length=1, color='k')
    ax.text(gravity_vector[0], gravity_vector[1], 1.2 * gravity_vector[2], "Gravity", color='k', size=15, zorder=1)

    # ---- Step 2: Draw the center of the apple ----
    a = b = c = 0
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    diam = 0.5
    x = (diam * np.outer(np.cos(u), np.sin(v))) + a
    y = (diam * np.outer(np.sin(u), np.sin(v))) + b
    z = (1.25 * diam * np.outer(np.ones(np.size(u)), np.cos(v))) + c
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='r', linewidth=0, alpha=0.2)
    ax.scatter(a, b, c, color="k", s=100)

    # ---- Step 1: Draw the Stem Vector ----
    calix = np.array([float(apple_calix[0]), float(apple_calix[1]), float(apple_calix[2])])
    stem = np.array([float(apple_stem[0]), float(apple_stem[1]), float(apple_stem[2])])
    stem_vector = np.subtract(stem, calix)
    # print("Before", stem_vector)
    stem_vector = stem_vector / np.linalg.norm(stem_vector)  # Normalize its magnitude
    # print("After", stem_vector)
    ax.quiver(0, 0, 0, stem_vector[0], stem_vector[1], stem_vector[2], length=1, color='brown')
    ax.text(stem_vector[0], stem_vector[1], stem_vector[2], "Stem", color='brown', size=15, zorder=1)

    # # ---- Step 2: Draw the gravity Vector ----
    # ax.quiver(0, 0, 0, 0, 0, -1, length=1, color='r')
    # ax.text(0, 0, -1, "Gravity", color='r', size=15, zorder=1)

    # # ---- Step 3: Draw the apple -----
    # a = float(apple_center[0])
    # b = float(apple_center[1])
    # c = float(apple_center[2])
    # a = b = c = 0
    #
    # u = np.linspace(0, 2 * np.pi, 100)
    # v = np.linspace(0, np.pi, 100)
    # diam = 0.5
    # x = (diam * np.outer(np.cos(u), np.sin(v))) + a
    # y = (diam * np.outer(np.sin(u), np.sin(v))) + b
    # z = (diam * np.outer(np.ones(np.size(u)), np.cos(v))) + c
    # ax.plot_surface(x, y, z, rstride=4, cstride=4, color='r', linewidth=0, alpha=0.2)

    # # ---- Step 4: Draw the center of the apple ----
    # ax.scatter(a, b, c, color="g", s=100)

    # ---- Step 5: Draw the Hand's axes
    hand_origin = np.array([float(hand_origin[0]), float(hand_origin[1]), float(hand_origin[2])])
    hand_x = np.array([float(hand_x[0]), float(hand_x[1]), float(hand_x[2])])
    hand_y = np.array([float(hand_y[0]), float(hand_y[1]), float(hand_y[2])])
    hand_z = np.array([float(hand_z[0]), float(hand_z[1]), float(hand_z[2])])
    hand_x_vector = np.subtract(hand_x, hand_origin)
    hand_x_vector = hand_x_vector / np.linalg.norm(hand_x_vector)
    # ax.quiver(0, 0, 0, hand_x_vector[0], hand_x_vector[1], hand_x_vector[2], length=1, color='k')
    # ax.text(hand_x_vector[0], hand_x_vector[1], hand_x_vector[2], "Hand Frame x", color='k', size=8, zorder=1)
    hand_y_vector = np.subtract(hand_y, hand_origin)
    hand_y_vector = hand_y_vector / np.linalg.norm(hand_y_vector)
    # ax.quiver(0, 0, 0, hand_y_vector[0], hand_y_vector[1], hand_y_vector[2], length=1, color='k')
    # ax.text(hand_y_vector[0], hand_y_vector[1], hand_y_vector[2], "Hand Frame y", color='k', size=8, zorder=1)
    hand_z_vector = np.subtract(hand_z, hand_origin)
    hand_z_vector = hand_z_vector / np.linalg.norm(hand_z_vector)
    ax.quiver(0, 0, 0, hand_z_vector[0], hand_z_vector[1], hand_z_vector[2], length=1, color='b')
    ax.text(hand_z_vector[0], hand_z_vector[1], hand_z_vector[2], "Hand", color='b', size=15, zorder=1)

    plt.suptitle("Hand and Stem w.r.t Base - Pick %i" % (k + 1))
    plt.title('Hand-Stem: %.0f\N{DEGREE SIGN} , Hand-Gravity: %.0f\N{DEGREE SIGN}, Stem-Gravity %0.f\N{DEGREE SIGN}' % (
        handToStem_angle, handToGravity_angle, stemToGravity_angle))


def draw_kmeans_in_base(i):
    hand_stem = np.radians(i[0])
    stem_gravity = np.radians(i[1])
    hand_gravity = np.radians(i[2])

    # ---- Step 0: Initialize Figure
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    size = 0.8
    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)
    ax.set_zlim(-size, size)
    ax.set_xlabel('World X')
    ax.set_ylabel('World Y')
    ax.set_zlabel('World Z')
    # Get rid of the ticks
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])

    ax.set_title('Hand-Stem= ' + str(int(i[0])) + '\N{DEGREE SIGN},\nStem-Gravity= ' + str(int(i[1])) + '\N{DEGREE SIGN},\nHand-Gravity= ' + str(int(i[2])) + '\N{DEGREE SIGN}')

    # ---- Step 1: Draw the gravity Vector ----
    gravity_vector = np.array([0, 0, -1])
    ax.quiver(0, 0, 0, gravity_vector[0], gravity_vector[1], gravity_vector[2], length=1, color='k')
    ax.text(gravity_vector[0], gravity_vector[1], 1.2 * gravity_vector[2], "Gravity", color='k', size=15, zorder=1)

    # ---- Step 2: Draw the center of the apple ----
    a = b = c = 0

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    diam = 0.5
    x = (diam * np.outer(np.cos(u), np.sin(v))) + a
    y = (diam * np.outer(np.sin(u), np.sin(v))) + b
    z = (1.25 * diam * np.outer(np.ones(np.size(u)), np.cos(v))) + c
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='r', linewidth=0, alpha=0.2)

    ax.scatter(a, b, c, color="k", s=100)

    # ---- Step 3: Draw the stem ----
    stem_vector = np.array([0, np.sin(stem_gravity), -np.cos(stem_gravity)])
    ax.quiver(0, 0, 0, stem_vector[0], stem_vector[1], stem_vector[2], length=1, color='brown')
    ax.text(stem_vector[0], stem_vector[1], stem_vector[2], "Stem", color='brown', size=15, zorder=1)

    # ---- Step 4: Draw the vector normal to hand ----
    hand_z = -np.cos(hand_gravity)
    hand_radius = np.sin(hand_gravity)
    for i in range(360):
        hand_x = hand_radius * np.cos(np.radians(i))
        hand_y = hand_radius * np.sin(np.radians(i))
        hand_prelim_vector = np.array([hand_x, hand_y, hand_z])
        hand_stem_prelim = np.arccos(np.dot(hand_prelim_vector, stem_vector))

        if abs(hand_stem_prelim - hand_stem) < 0.005:
            hand_vector = hand_prelim_vector

    ax.quiver(0, 0, 0, hand_vector[0], hand_vector[1], hand_vector[2], length=1, color='blue')
    ax.text(hand_vector[0], hand_vector[1], hand_vector[2], "Hand", color='blue', size=15, zorder=1)

    set_axes_equal(ax)



    # ---- Step 5: Check that the angles are ok ----
    # print('\nChecking angles')
    print('\nStem-Gravity angles', np.degrees(np.arccos(np.dot(stem_vector, gravity_vector))))
    print('Hand-Gravity angles', np.degrees(np.arccos(np.dot(hand_vector, gravity_vector))))
    print('Hand-Stem angles', np.degrees(np.arccos(np.dot(hand_vector, stem_vector))))


if __name__ == '__main__':

    # ---------------------------------------- Step 1 - Read the csv files ---------------------------------------------
    location = 'C:/Users/15416/PycharmProjects/PickApp/data/'
    # Read the csv file with all the coordinates in the hand's coordinate frame
    file = 'objects_in_hand.csv'
    with open(location + file, 'r') as f:
        reader = csv.reader(f)
        apple_coords = list(reader)
    apples = len(apple_coords)
    print('\nThe number of apples were:', len(apple_coords))

    # Read the csv file with all the coordinates transformed into the baselink
    file = 'objects_in_base.csv'
    with open(location + file, 'r') as f:
        reader = csv.reader(f)
        apple_coords_base = list(reader)

    # --------------------------------------- Step 2 - Sweep all the coordinates and plot ------------------------------

    hand_to_stem_angles = []
    hand_to_gravity_angles = []
    stem_to_gravity_angles = []

    success_hand_to_stem_angles = []
    success_hand_to_gravity_angles = []
    success_stem_to_gravity_angles = []

    angles_for_kmeans = []

    failed_hand_to_stem_angles = []
    failed_hand_to_gravity_angles = []
    failed_stem_to_gravity_angles = []

    # Successful real apple picks
    success_picks = [6, 10, 16, 30, 31, 38, 42, 43, 48, 50, 51, 52, 53, 60, 61, 63, 64, 67, 70, 71, 72, 73, 74, 77]

    for k in range(apples):

        i = apple_coords[k]
        j = apple_coords_base[k]

        handToStem_angle, handToGravity_angle, stemToGravity_angle = draw_in_hand(i, False)   # Plot in hand's c-frame
        draw_in_base(j, handToStem_angle, handToGravity_angle, stemToGravity_angle)     # Plot in baselink's c-frame

        if (k+1) in success_picks:
            # Save it in success angles
            success_hand_to_stem_angles.append(handToStem_angle)
            success_hand_to_gravity_angles.append(handToGravity_angle)
            success_stem_to_gravity_angles.append(stemToGravity_angle)

        else:
            # Save it in failed angles
            failed_hand_to_stem_angles.append(handToStem_angle)
            failed_hand_to_gravity_angles.append(handToGravity_angle)
            failed_stem_to_gravity_angles.append(stemToGravity_angle)

        hand_to_stem_angles.append(handToStem_angle)
        hand_to_gravity_angles.append(handToGravity_angle)
        stem_to_gravity_angles.append(stemToGravity_angle)

        angles_for_kmeans.append([handToStem_angle, stemToGravity_angle, handToGravity_angle])

    # --------------------------------------- Step 3 - Do Plots --------------------------------------------------------
    # Boxplots
    angles = [hand_to_stem_angles, stem_to_gravity_angles, hand_to_gravity_angles]
    fig, ax = plt.subplots()
    ax.boxplot(angles)
    ax.set_xticklabels(['Hand-Stem', 'Stem-Gravity', 'Hand-Gravity'])
    ax.set_xlabel('Real apple pick angles')
    ax.set_ylabel('Angle [deg]')
    ax.yaxis.grid()

    # Scatterplot
    fig = plt.figure()

    ax = plt.axes(projection='3d')
    ax.scatter(success_hand_to_stem_angles, success_stem_to_gravity_angles, success_hand_to_gravity_angles,
               alpha=0.5, s=20, c='g', depthshade=False, label='Successful picks')
    ax.scatter(failed_hand_to_stem_angles, failed_stem_to_gravity_angles, failed_hand_to_gravity_angles,
               alpha=0.5, s=20, c='r', depthshade=False, label='Failed picks')
    ax.set_xlabel('Hand-Stem [deg]', fontsize=15)
    ax.set_ylabel('Stem-Gravity [deg]', fontsize=15)
    ax.set_zlabel('Hand-Gravity [deg]', fontsize=15)
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)
    plt.setp(ax.get_zticklabels(), fontsize=12)
    ax.set_title('Real apple pick angles')


    # K-means Scatter plot
    kmeans = KMeans(n_clusters=5, random_state=0).fit(angles_for_kmeans)
    print(kmeans.cluster_centers_)
    alpha = []
    beta = []
    gamma = []
    for i in kmeans.cluster_centers_:
        alpha.append(i[0])
        beta.append(i[1])
        gamma.append(i[2])
        draw_kmeans_in_base(i)

    # ax.minorticks_on()
    # ax.scatter(alpha, beta, gamma, alpha=1, s=80, c='k', marker='^', depthshade=False, label='k-means')

    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    # Customize the minor grid
    # ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    ax.legend()

    plt.show()

    # Show the angles of a specific pick
    target = 50 - 1
    print('The angles are', angles_for_kmeans[target])

    # ---------------------------------------- Step 3 - Store angles in a csv file -------------------------------------
    with open('../data/angles.csv', 'w') as f:
        write = csv.writer(f)
        write.writerows(angles)
