import glob
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ...classification import ClsfEvaluation
sns.set()
# sns.set_style("whitegrid")
# sns.set_style("darkgrid")

def check_pass(df, option, result_img_dir):
    pass_list = []
    for img_dir in df['image']:
        img_file = os.path.split(img_dir)[-1]
        img_name = os.path.splitext(img_file)[0]

        search_img_dir = os.path.join(result_img_dir, 
                                      img_name + '_{}*'.format(option))
        
        available_files = glob.glob(search_img_dir)

        if len(available_files) == 0:
            pass_list.append(0)
        else:
            status = None
            for each_file in available_files:

                file_name = os.path.split(each_file)[-1]
                name_ = os.path.splitext(file_name)[0]
                if name_[-5:] == 'limit':
                    status = 2
                elif name_[-9:] in ['lastpoint', 'deadpoint']:
                    status = 1
            pass_list.append(status)
    return pass_list


def plot_stats(df, option, x_lable, lim, message_):
    status_name = {0: 'Not started', 1: 'In-progress', 2: 'Done'}
    g = sns.displot(data=df, 
                    x=option, 
                    # kde=True, 
                    rug=True, 
                    height=6, 
                    aspect=6/6, 
                    hue=df["{} status".format(option)].map(status_name), 
                    hue_order=list(status_name.values()),
                    multiple="stack")
    new_label = x_lable + ' ({})'.format(message_)

    # Change labels
    g.set_axis_labels(new_label, "count")
    # g.set(title = "blurring")

    # show count on the top of bar
    for ax in g.axes.flat:
        # print(ax.containers)
        ax.bar_label(ax.containers[-1])
        # ax.bar_label(sum(ax.containers))

    # set limit range for x and y
    if option in lim.keys():
        value = lim.get(option)
        plt.xlim(*value)

    # miscellaneous
    des_df = df.describe()
    title = ("Min = {:.2f}, Max = {:.2f}, Mean = {:.2f}".format(des_df[option]["min"],
                                                                des_df[option]["max"],
                                                                des_df[option]["mean"]))
    plt.title(title)
    g.fig.set_dpi(100)
    plt.show()

# def multiplot(df, option):
#     status_name = {0: 'Not started', 1: 'In-progress', 2: 'Done'}
#     # Creating subplot axes
#     fig, axes = plt.subplots(2, 2)

#     for i, opt, ax in zip(range(len(option)), option, axes.flatten()):
#         sns.histplot(data=df, 
#                      x=opt, 
#                      # kde=True, 
#                      # rug=True, 
#                      hue=df["{} status".format(opt)].map(status_name), 
#                      hue_order=list(status_name.values()),
#                      multiple="stack",
#                      ax=ax)
#         if i > 0:
#             if ax is None:
#                 continue
#             else:
#                 print(ax)
#                 ax.get_legend().remove()

#     # # Iterating through axes and names
#     # for name, ax in zip(names, axes.flatten()):
#     #     sns.boxplot(y=name, x= "a", data=df, orient='v', ax=ax)
#     plt.show()


LIM = {
    # 'blurring': 
    'increasing_brightness': [0.0, 1.0],
    'increasing_contrast': [0.0, 255.0],
    'decreasing_brightness': [0, -1],
    'decreasing_contrast': [0, -1],
    'down_scale': [1.0, 0.0],
    'crop': [5/5, 5/16],
    'left_rotation': [0, 45],
    'right_rotation': [0, -45],
    'compactness': [1.0, 0],
}

X_LABEL = [
    'increasing brightness',
    'increasing contrast',
    'decreasing brightness',
    'decreasing contrast',
    'down scale',
    'crop',
    'left rotation',
    'right rotation',
    'compactness',
]

option_list = list(LIM.keys())

instance_ = ClsfEvaluation('foo', 'bar')

df_first = pd.read_csv('result_folder/result.csv')
df_last = pd.read_csv('rc_result_folder/result.csv')

# print("Histogram")
# for i in range(len(option_list)):
#     opt = option_list[i]
#     mess_ = instance_.report[opt]['note']
#     if i < 6:
#         new_column_name = opt + " status"
#         df_first[new_column_name] = check_pass(df_first, opt, "result_folder/images")
#         plot_stats(df_first, opt, X_LABEL[i], LIM, mess_)
#         print(df_first.head(10))
#     else:
#         new_column_name = opt + " status"
#         df_last[new_column_name] = check_pass(df_last, opt, "rc_result_folder/images")
#         plot_stats(df_last, opt, X_LABEL[i], LIM, mess_)


