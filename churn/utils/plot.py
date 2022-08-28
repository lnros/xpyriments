"""
Plot Utils
"""

from datetime import datetime
from itertools import cycle
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
from numpy import interp
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, auc, f1_score, \
    confusion_matrix, PrecisionRecallDisplay, average_precision_score
from typing import AnyStr, NoReturn, Optional, List, Union, Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

REPO_PATH = Path(__file__).parent.parent.resolve()
SAVE_PATH = os.path.join(REPO_PATH, 'images')


class EDA:
    """
    Plots for performing Exploratory Data Analysis
    """

    def __init__(self, font_scale=1.0, fig_size=(15, 10), save=False,
                 save_path=SAVE_PATH):
        self.font_scale = font_scale
        self.fig_size = fig_size
        self.save = save
        self.save_path = save_path

    def barplot_normalized(self, df: pd.DataFrame,
                           feature_name: AnyStr,
                           class_name: AnyStr,
                           rotate: bool = False,
                           min_obs_to_keep: int = 50) -> NoReturn:
        """
        Displays a normalized bar plot for each category of
                 the chosen feature for each output of the class variable
        :param min_obs_to_keep: min observations of a category to keep it for
                                visualization
        :param df: input dataframe
        :param feature_name: col name of a categorical variable
        :param class_name: col name of the class variable
        :param rotate: indicates whether to present the x ticks in
                       90 degrees rotation
        """
        df_to_plot = df.copy()
        categories_to_drop = [category for category in
                              pd.unique(df_to_plot[feature_name]) if
                              df_to_plot[feature_name].value_counts()[
                                  category] < min_obs_to_keep]
        df_to_plot = (df_to_plot.groupby([feature_name])[class_name]
                      .value_counts(normalize=True)
                      .rename('percentage')
                      .mul(100)
                      .reset_index()
                      .sort_values(class_name))
        df_to_plot = df_to_plot[
            ~df_to_plot[feature_name].isin(categories_to_drop)]
        mapping = {category: str(category) + ': ' + str(
            list(df[feature_name].value_counts())[ind]) for ind, category in
                   enumerate(df[feature_name].value_counts().index)}
        df_to_plot = df_to_plot.replace({feature_name: mapping})
        sns.set(font_scale=self.font_scale)
        sns.barplot(x=feature_name, y='percentage', hue=class_name,
                    data=df_to_plot)
        plt.xticks(rotation=90 * rotate)
        if self.save:
            now = datetime.now().strftime('%Y_%m_%d_%H_%M')
            path = os.path.join(self.save_path, 'barplot')
            if not os.path.isdir(path):
                os.mkdir(path)
            filename = os.path.join(path, f'barplot_{now}')
            i = 0
            while os.path.exists(f'{filename}_{i:d}.png'):
                i += 1
            plt.savefig(f'{filename}_{i}.png')
        plt.show()

    def boxplot_normalized(self, df: pd.DataFrame,
                           feature_name: AnyStr,
                           class_name: AnyStr,
                           target_name: AnyStr,
                           rotate: bool = False,
                           min_obs_to_keep: int = 50) -> NoReturn:
        """
        Displays a normalized boxplot for each category of
        the chosen feature for each output of the target variable
        :param min_obs_to_keep: min observations of a category to keep it for
                                visualization
        :param df: input dataframe
        :param feature_name: col name of a categorical variable
        :param target_name: col name of the target variable
        :param rotate: indicates whether to present the x ticks in
                       90 degrees rotation
        """
        df_to_plot = df.copy()
        categories_to_drop = [category for category in
                              pd.unique(df_to_plot[feature_name]) if
                              df_to_plot[feature_name].value_counts()[
                                  category] < min_obs_to_keep]
        df_to_plot = (
            df_to_plot.groupby([class_name, feature_name])[target_name]
                .value_counts(normalize=True)
                .rename('percentage')
                .mul(100)
                .reset_index()
                .sort_values(target_name))
        df_to_plot = df_to_plot[
            ~df_to_plot[feature_name].isin(categories_to_drop)]
        mapping = {category: str(category) + ': ' + str(
            list(df[feature_name].value_counts())[ind]) for ind, category in
                   enumerate(df[feature_name].value_counts().index)}
        df_to_plot = df_to_plot.replace({feature_name: mapping})
        sns.set(font_scale=self.font_scale)
        sns.boxplot(x=feature_name, y=target_name, hue=class_name,
                    data=df_to_plot)
        plt.xticks(rotation=90 * rotate)
        if self.save:
            now = datetime.now().strftime('%Y_%m_%d_%H_%M')
            path = os.path.join(self.save_path, 'boxplot')
            if not os.path.isdir(path):
                os.mkdir(path)
            filename = os.path.join(path, f'boxplot_{now}')
            i = 0
            while os.path.exists(f'{filename}_{i:d}.png'):
                i += 1
            plt.savefig(f'{filename}_{i}.png')
        plt.show()

    def pareto(self,
               df: pd.DataFrame,
               groupby: str,
               aggregations: Dict[str, Any]):
        """
        :param df:
        :param groupby:
        :param aggregations:
        :return:
        """
        sum_col, count_col = '', ''
        for col, aggs in aggregations.items():
            if 'count' in aggs or 'count' == aggs:
                count_col = col
            if 'sum' in aggs or 'sum' == aggs:
                sum_col = col

        clean_name_agg_col = ' '.join(sum_col.split('_')).title()
        clean_name_gb = ' '.join(groupby.split('_')).title()

        df_pareto = df.groupby(groupby).agg(aggregations)
        df_pareto['count_pct'] = df_pareto.loc[:,
                                 (count_col, 'count')] / df_pareto.loc[:,
                                                         (count_col,
                                                          'count')].sum()
        df_pareto['sum_pct'] = df_pareto.loc[:,
                               (sum_col, 'sum')] / df_pareto.loc[:,
                                                   (sum_col, 'sum')].sum()
        df_pareto = df_pareto.sort_values('sum_pct', ascending=False)
        df_pareto['count_pct_cumsum'] = df_pareto['count_pct'].cumsum()
        df_pareto['sum_pct_cumsum'] = df_pareto['sum_pct'].cumsum()
        df_pareto = df_pareto.reset_index()
        sns.set(font_scale=self.font_scale)
        fig, ax = plt.subplots(figsize=self.fig_size)
        bars = sns.barplot(x=df_pareto[groupby],
                           y=df_pareto.loc[:, sum_col]['sum'],
                           ax=ax,
                           alpha=0.7,
                           color='green',
                           label=f'Total {sum_col}')
        annotate_bars(bars)
        ax.grid(False)
        ax2 = ax.twinx()
        ax2.grid(False)
        sns.lineplot(data=df_pareto,
                     x=groupby,
                     y='sum_pct_cumsum',
                     marker='o',
                     lw=2,
                     ax=ax2,
                     color='black',
                     label=f"Cumulative Percentage ({clean_name_agg_col})")

        xs = df_pareto.index
        ys = df_pareto['sum_pct_cumsum']
        if len(xs) > 5:
            ax2.set_xticks(df_pareto[groupby].values, rotation=45)
        annotate_line(xs, ys, xytext=(0, 10))

        sns.lineplot(data=df_pareto,
                     x=groupby,
                     y='count_pct_cumsum',
                     marker='o',
                     lw=2,
                     ax=ax2,
                     color='magenta',
                     label='Cumulative Percentage (N Cases)')
        ys = df_pareto['count_pct_cumsum']
        annotate_line(xs, ys, xytext=(0, -15))

        plt.title(f'Pareto: {clean_name_gb}', fontsize=20)
        ax.set_ylabel(f'Total {clean_name_agg_col}')
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax2.set_ylabel('Cumulative percentage')
        ax2.set_yticks(np.arange(0, 1.1, 0.1))
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))

        plt.legend(loc='lower left')
        if self.save:
            now = datetime.now().strftime('%Y_%m_%d_%H_%M')
            path = os.path.join(self.save_path, 'pareto')
            if not os.path.isdir(path):
                os.mkdir(path)
            filename = os.path.join(path, f'pareto_{now}')
            i = 0
            while os.path.exists(f'{filename}_{i:d}.png'):
                i += 1
            plt.savefig(f'{filename}_{i}.png')
        plt.show()


class Evaluation:
    """
    Plots relevant for evaluating a model
    """

    def __init__(self, font_scale: float = 1.0, fig_size=(24, 18),
                 save=False, show=True) -> NoReturn:
        self.font_scale = font_scale
        self.fig_size = fig_size
        self.save = save
        self.show = show

    def plot_feature_importance(self, feature_importance: pd.DataFrame,
                                to_save: bool = False,
                                n_features_to_show: Optional[int] = None,
                                ) -> NoReturn:
        """
        :param feature_importance: df consist of the feature names in the
                                   first columns and the importance
                                   for each fold
        :param: to_save: whether to save the plot as a png file
        :param n_features_to_show:
        """
        n_folds = feature_importance.shape[1] - 1
        feature_importance['average_importance'] = feature_importance[
            [f'fold_{fold_n + 1}' for fold_n in range(n_folds)]].mean(axis=1)

        n_features_to_show = feature_importance.shape[0] \
            if n_features_to_show is None else n_features_to_show

        plt.figure(figsize=self.fig_size)
        sns.set(font_scale=self.font_scale)
        input_data = feature_importance.sort_values(
            by='average_importance',
            ascending=False) \
            .head(n_features_to_show)
        sns.barplot(data=input_data, x='average_importance', y='feature')
        title = f'Feature Importance over {n_folds} folds'
        plt.title(title)
        if self.save:
            filename = "_".join(title.lower().split()) + '.png'
            save_path = os.path.join(
                SAVE_PATH, filename
            )
            plt.savefig(save_path, bbox_inches='tight')
        if self.show:
            plt.show()
        else:
            plt.close()

    def plot_cv_precision_recall(self,
                                 clf: Any,
                                 n_folds: int,
                                 n_repeats: int,
                                 X: pd.DataFrame,
                                 y: pd.Series,
                                 random_state: int = 0,
                                 stacking: bool = False,
                                 title: AnyStr = "") -> pd.DataFrame:
        """
        Plots the Precision-Recall curve and returns the feature importance.
        :param clf: A tree classifier
        :param n_folds: number of folds for cross validation
        :param n_repeats:
        :param X: input data
        :param y: target data
        :param random_state: random state seed
        :param stacking: whether stacking is used to create the classifier
        :param title: plot's title
        :return: feature importance
        """

        sns.set(font_scale=self.font_scale)
        cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats,
                                     random_state=random_state)
        n_classes = len(y.unique())
        is_multiclass = n_classes > 2
        f1s = []
        prs = []
        aucs = []
        mean_recall = np.linspace(0, 1, 100)
        feature_importance = pd.DataFrame({'feature': X.columns})
        if not is_multiclass:
            plt.figure(figsize=self.fig_size)
        i = 0
        for train, test in cv.split(X, y):
            x_train, x_test = X.loc[X.index[train], :], X.loc[X.index[test], :]
            y_train, y_test = y.index[train], y.index[test]
            probas_ = clf.fit(x_train, y_train).predict_proba(x_test)
            y_pred = np.argmax(probas_, axis=1)
            feature_importance = self._get_feature_importance_per_fold(
                feature_importance, i, clf, stacking
            )
            # Compute PR curve and area under the curve
            if not is_multiclass:
                precision, recall, thresholds = precision_recall_curve(
                    y[y.index[test]], probas_[:, 1])
                prs.append(interp(mean_recall, precision, recall))
                pr_auc = auc(recall, precision)
                aucs.append(pr_auc)
                f1s.append(f1_score(y_pred=y_pred, y_true=y[test],
                                    average='weighted'))
                plt.plot(recall, precision, lw=3, alpha=0.5,
                         label='Fold %d (AUCPR = %0.2f)' % (i + 1, pr_auc))
            i += 1

        if is_multiclass:
            self._plot_multiclass_precision_recall_curve(X, y, clf, n_classes,
                                                         random_state, title)
        else:
            plt.hlines(y=np.mean(y), xmin=0.0, xmax=1.0, linestyle='--', lw=3,
                       color='k', label='Luck', alpha=.8)
            mean_precision = np.mean(prs, axis=0)
            mean_auc = auc(mean_recall, mean_precision)
            std_auc = np.std(aucs)
            self._plot_binary_precision_recall_curve(mean_precision,
                                                     mean_recall, mean_auc,
                                                     std_auc, f1s, title)

        return feature_importance

    def plot_confusion_matrix(self, y_test, y_pred):
        sns.set(font_scale=self.font_scale)
        cf_matrix = confusion_matrix(y_test, y_pred)
        group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        group_counts = ["{0:0.0f}".format(value) for value in
                        cf_matrix.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in
                             cf_matrix.flatten() / np.sum(cf_matrix)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                  zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
        if self.save:
            filename = 'cm.png'
            save_path = os.path.join(
                SAVE_PATH, filename
            )
            plt.savefig(save_path, bbox_inches='tight')
        if self.show:
            plt.show()
        else:
            plt.close()

    def plot_precision_recall_test(self, y_true, y_scores, title=''):
        mean_recall = np.linspace(0, 1, 100)
        precision, recall, thresholds = precision_recall_curve(y_true,
                                                               y_scores)
        mean_precision = interp(mean_recall, precision, recall)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=self.fig_size)
        sns.set(font_scale=self.font_scale)
        plt.hlines(y=np.mean(y_true), xmin=0.0, xmax=1.0, linestyle='--', lw=3,
                   color='k', label='Luck', alpha=.8)
        plt.plot(mean_precision, mean_recall, color='navy',
                 label=r'(AUCPR =' + str(np.round(pr_auc, 2)), lw=4)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.title(title,
                  fontdict={'family': 'serif', 'color': 'darkred', 'size': 50})
        plt.xlabel('Recall', fontweight="bold", fontsize=30)
        plt.ylabel('Precision', fontweight="bold", fontsize=30)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.legend(prop={'size': 20}, loc=0)
        if self.save:
            filename = "_".join(title.lower().split()) + '.png'
            save_path = os.path.join(
                SAVE_PATH, filename
            )
            plt.savefig(save_path, bbox_inches='tight')
        if self.show:
            plt.show()
        else:
            plt.close()

    def _plot_multiclass_precision_recall_curve(self, X, y, clf, n_classes,
                                                random_state=0, title=None):

        luck = y.value_counts(normalize=True)
        y = label_binarize(y, classes=np.arange(n_classes))

        # Split into training and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=random_state
        )

        clf = OneVsRestClassifier(clf)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                                y_score[:, i])
            average_precision[i] = average_precision_score(y_test[:, i],
                                                           y_score[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_test.ravel(), y_score.ravel()
        )
        average_precision["micro"] = average_precision_score(y_test, y_score,
                                                             average="micro")

        # setup plot details
        colors = cycle(
            ["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

        _, ax = plt.subplots(figsize=self.fig_size)

        f_scores = np.linspace(0.2, 0.8, num=4)

        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y_ = f_score * x / (2 * x - f_score)
            (l,) = plt.plot(x[y_ >= 0], y_[y_ >= 0], color="gray", alpha=0.2)
            plt.annotate(f"f1={f_score:.1f}", xy=(0.9, y_[45] + 0.02),
                         fontsize=20)

        display = PrecisionRecallDisplay(
            recall=recall["micro"],
            precision=precision["micro"],
            average_precision=average_precision["micro"],
        )
        display.plot(ax=ax,
                     name="Micro-average precision-recall",
                     color="gold")

        for i, color in zip(range(n_classes), colors):
            display = PrecisionRecallDisplay(
                recall=recall[i],
                precision=precision[i],
                average_precision=average_precision[i],
            )
            display.plot(ax=ax,
                         name=f"Precision-recall for class {i}",
                         color=color)

            plt.hlines(y=luck, xmin=0.0, xmax=1.0, linestyle='--', lw=3,
                       color='k', alpha=.8)
            x_pos = 0.3 + (i * 0.125)
            plt.annotate(f'Luck (class {i})', xy=(x_pos,
                                                  luck[i] + 0.02),
                         fontsize=20)

        # add the legend for the iso-f1 curves
        handles, labels = display.ax_.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["iso-f1 curves"])
        # set the legend and the axes
        ax.set_xlabel('Recall', fontweight="bold", fontsize=30)
        ax.set_ylabel('Precision', fontweight="bold", fontsize=30)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(handles=handles, labels=labels, loc="best", fontsize=20)
        if title is None:
            title = "Precision-Recall curve to multi-class"
        ax.set_title(title,
                     fontdict={'family': 'serif', 'color': 'darkred',
                               'size': 50})
        plt.tick_params(axis='both', which='major', labelsize=20)
        if self.save:
            filename = "_".join(title.lower().split()) + '.png'
            save_path = os.path.join(
                SAVE_PATH, filename
            )
            plt.savefig(save_path, bbox_inches='tight')
        if self.show:
            plt.show()
        else:
            plt.close()

    def _plot_binary_precision_recall_curve(self,
                                            mean_precision,
                                            mean_recall,
                                            mean_auc,
                                            std_auc,
                                            f1s,
                                            title=None):
        plt.plot(mean_precision, mean_recall, color='navy',
                 label=r'Mean (AUCPR = %0.3f $\pm$ %0.2f)' % (
                     mean_auc, std_auc), lw=4)
        plt.title(title,
                  fontdict={'family': 'serif', 'color': 'darkred', 'size': 50})
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall', fontweight="bold", fontsize=30)
        plt.ylabel('Precision', fontweight="bold", fontsize=30)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.legend(prop={'size': 20}, loc=0)
        plt.text(x=0.2, y=0.3, s=r'Mean (F1 = %0.3f $\pm$ %0.2f)' % (
            np.mean(f1s), np.std(f1s)), fontsize=20)
        if self.save:
            filename = "_".join(title.lower().split()) + '.png'
            save_path = os.path.join(
                SAVE_PATH, filename
            )
            plt.savefig(save_path, bbox_inches='tight')
        if self.show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def _get_feature_importance_per_fold(feature_importance,
                                         fold_number,
                                         clf,
                                         stacking):

        if not stacking:
            feature_importance[
                f'fold_{fold_number + 1}'] = clf.feature_importances_
        else:
            weights = clf.final_estimator_.coef_
            weights /= np.sum(weights)
            imp = np.zeros((feature_importance.shape[0], len(clf.estimators_)))
            for ind, est in enumerate(clf.estimators_):
                clf_i = clf.estimators_[ind]
                imp_i = clf_i.feature_importances_
                imp[:, ind] = (imp_i - np.min(imp_i)) / (
                        np.max(imp_i) - np.min(imp_i))
            avg_imp = np.matmul(imp, np.transpose(weights))
            feature_importance[f'fold_{fold_number + 1}'] = avg_imp
        return feature_importance


def annotate_bars(bar_plot):
    for p in bar_plot.patches:
        bar_plot.annotate(f'{p.get_height():,.0f}',
                          (p.get_x() + p.get_width() / 2,
                           p.get_height()), ha='center', va='center',
                          size=15, xytext=(0, 8),
                          textcoords='offset points')


def annotate_line(xs, ys, xytext):
    for x, y in zip(xs, ys):
        label = f"{y:.0%}"
        plt.annotate(label,  # this is the text
                     (x, y),
                     # these are the coordinates to position the label
                     textcoords="offset points",
                     # how to position the text
                     xytext=xytext,
                     # distance from text to points (x,y)
                     ha='center')
