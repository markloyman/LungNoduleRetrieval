from init import *
from experiments import load_experiments
from Analysis.performance import eval_retrieval, eval_classification, eval_correlation


def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{:.3}'.format(height).lstrip('0'), ha=ha[xpos], va='bottom')


if __name__ == "__main__":

    # Setup
    exp_name = 'DirRating'
    dset = 'Test'
    rating_norm = 'none'
    start = timer()
    num_of_indexes = 3 + 4

    runs, run_net_types, run_metrics, _, run_names, run_ep_perf, run_ep_comb = load_experiments(exp_name)

    data = np.zeros((len(runs), num_of_indexes))
    dataStd = np.zeros((len(runs), num_of_indexes))

    # evaluate
    run_id = 0
    for run, net_type, _, metric, epochs in zip(runs, run_net_types, range(len(runs)), run_metrics, run_ep_comb):
        print("Evaluating classification accuracy for {}{}".format(net_type, run))
        acc, acc_std, _ = eval_classification(
            run=run, net_type=net_type, dset=dset,
            metric=metric, epochs=epochs,
            cross_validation=True)
        print("Evaluating retrieval precision for {}{}".format(net_type, run))
        prec, prec_std, index, index_std, _ = eval_retrieval(
            run=run, net_type=net_type, dset=dset,
            metric=metric, epochs=epochs,
            cross_validation=True)

        data[run_id, 0] = acc
        data[run_id, 1] = prec
        data[run_id, 2] = index

        dataStd[run_id, 0] = acc_std
        dataStd[run_id, 1] = prec_std
        dataStd[run_id, 2] = index_std

        pm, pm_std, km, km_std, pr, pr_std, kr, kr_std, _ = eval_correlation(
            run=run, net_type=net_type, dset=dset, rating_norm=rating_norm,
            metric=metric, epochs=epochs,
            cross_validation=True)

        data[run_id, 3] = pm
        data[run_id, 4] = pr
        data[run_id, 5] = km
        data[run_id, 6] = kr

        dataStd[run_id, 3] = pm_std
        dataStd[run_id, 4] = pr_std
        dataStd[run_id, 5] = km_std
        dataStd[run_id, 6] = kr_std

        run_id += 1

    print("Evaluation Done in {:.1f} hours".format((timer() - start) / 60 / 60))

    ind = np.arange(num_of_indexes)  # the x locations for the groups
    width = 1/(len(runs)+1)  # the width of the bars

    fig, ax = plt.subplots()
    run_colors = ['SkyBlue', 'IndianRed', 'ForestGreen', 'Orchid', 'Orange']
    for run_id, name in enumerate(run_names):
        rects = ax.bar(ind - width / 2 + run_id * width, data[run_id], width, yerr=dataStd[run_id],
                       color=run_colors[run_id % len(run_colors)], label=name)
        autolabel(rects, 'left')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Test Report: ' + exp_name)
    ax.set_xticks(ind)
    ax.grid(which='both', axis='y')
    ax.set_xticklabels(('Acc', 'Prec', 'RetIdx', 'P-malig', 'P-rating', 'K-malig', 'K-rating'))
    ax.legend()

    plt.show()
