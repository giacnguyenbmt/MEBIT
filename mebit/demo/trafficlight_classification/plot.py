import glob
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from ...classification import ClsfEvaluation
sns.set()

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
                elif name_[-11:] == 'not-started':
                    status = 0
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
    new_label = x_lable + '\n({})'.format(message_)

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
    plt.subplots_adjust(left=None, bottom=0.13, right=None, top=0.95)
    g.fig.set_dpi(100)
    plt.show()
    # plt.savefig("{}_count.jpg".format(option))


def plot_metric_limit(df, x_lable, lim, message_):
    plt.figure(figsize=(8,6))
    if "accuracy" in df.columns:
        df = df.drop(columns=['accuracy'])

    cols = df.columns
    option = cols[0].replace(' limit', '')

    x = df[cols[0]].to_numpy()
    for i in range(1, len(cols)):
        y = df[cols[i]].to_numpy()
        plt.plot(x, y, drawstyle='steps', label=cols[i])

    plt.grid(axis='x', color='0.7')
    plt.legend(title="Metrics")
    new_label = x_lable + '\n({})'.format(message_)
    plt.xlabel(new_label)
    plt.ylabel("Metric value")

    # set limit range for x and y
    if option in lim.keys():
        value = lim.get(option)
        plt.xlim(*value)
    
    plt.show()
    # plt.savefig("{}_metric.jpg".format(option))


def get_label_at_limit(df, idx, unique_limit, result_dir, opt):
    dt = []
    gt = []
    dt_bool = []
    for j in range(len(df)):
        if df[opt].iloc[j] in unique_limit[:idx + 1]:
            dt_bool.append(True)
        else:
            dt_bool.append(False)

    for i in range(len(df)):
        record_ = df.iloc[i]
        status = record_['{} status'.format(opt)]
        image_file = os.path.split(record_.image)[-1]
        image_name = os.path.splitext(image_file)[0]
        
        file_ = "{}_{}_{}.txt".format(image_name, opt, STATUS_TYPE[status])
        if dt_bool[i] == True:
            dt_file_path = os.path.join(result_dir, "dt", file_)
        else:
            dt_file_path = os.path.join(result_dir, "gt", file_)

        with open(dt_file_path) as f:
            value = f.read()
        dt.append(value)

        gt_file_path = os.path.join(result_dir, "gt", file_)
        
        with open(gt_file_path) as f:
            value = f.read()
        gt.append(value)

    return gt, dt


def calculate_metric_by_limit(df, opt, result_dir, eval_func, ascending=True):
    df_ = df[["image", opt, '{} status'.format(opt)]]
    df_ = df_.sort_values(by=['{} status'.format(opt), opt], ascending=[True, ascending])
    unique_limit = np.sort(df_[opt].unique())[::1 if ascending else -1]

    metrics = []
    for i, thresh in enumerate(unique_limit):
        gt, dt = get_label_at_limit(df_, i, unique_limit, result_dir, opt)
        metric = eval_func(gt, dt)
        metrics.append(metric)

    list_of_keys = list(metrics[0].keys())
    formated_metrics = {}
    for k in list_of_keys:
        formated_metrics[k] = [value[k] for value in metrics]
    
    d = {'{} limit'.format(opt): unique_limit, **formated_metrics}
    result_df = pd.DataFrame(data=d)
    return result_df


LIM = {
    'blurring': [0, 215],
    'increasing_brightness': [0.0, 1.0],
    'increasing_contrast': [0.0, 255.0],
    'decreasing_brightness': [0, -1],
    'decreasing_contrast': [0, -255.0],
    'down_scale': [1.0, 0.0],
    'crop': [5/5, 5/16],
    'left_rotation': [0, 45],
    'right_rotation': [0, -45],
    'compactness': [1.0, 0],
}
X_LABEL = [
    'blurring limit',
    'brightness limit',
    'contrast limit',
    'brightness limit',
    'contrast limit',
    'scale ratio limit',
    'crop alpha limit',
    'rotation limit',
    'rotation limit',
    'compactness ratio limit (traffic_light_area / image_area)',
]
STATUS_TYPE = {
    0: "not-started",
    1: "deadpoint",
    2: "limit"
}
option_list = list(LIM.keys())

instance_ = ClsfEvaluation('foo', 'bar')

result_dir = 'result_folder'
rc_result_dir = 'rc_result_folder'

df_first = pd.read_csv(os.path.join(result_dir, 'result.csv'))
df_last = pd.read_csv(os.path.join(rc_result_dir, 'result.csv'))

print("num image x limit")
for i in range(len(option_list)):
    opt = option_list[i]
    mess_ = instance_.report[opt]['note']
    if i < 7:
        new_column_name = opt + " status"
        print(opt, X_LABEL[i], LIM[opt])
        df_first[new_column_name] = check_pass(df_first, opt, os.path.join(result_dir, "images"))
        plot_stats(df_first, opt, X_LABEL[i], LIM, mess_)
    else:
        new_column_name = opt + " status"
        df_last[new_column_name] = check_pass(df_last, opt, os.path.join(rc_result_dir, "images"))
        plot_stats(df_last, opt, X_LABEL[i], LIM, mess_)

print("metric x limit")
for i in range(len(option_list)):
    if i < 7:
        continue
    # if i < 8:
    #     continue
    opt = option_list[i]
    mess_ = instance_.report[opt]['note']
    limit_order = True if mess_ == "higher is better" else False

    result_df = calculate_metric_by_limit(df_last, opt, rc_result_dir, instance_.evaluate, limit_order)
    plot_metric_limit(result_df, X_LABEL[i], LIM, mess_)