import csv
from multiprocessing import Process
import sys

import numpy as np

from .ccn import convnet
from .ccn import options
from .script import get_sections
from .script import run_model


def make_predictions(net, data, labels, num_classes):
    data = np.require(data, requirements='C')
    labels = np.require(labels, requirements='C')

    preds = np.zeros((data.shape[1], num_classes), dtype=np.single)
    softmax_idx = net.get_layer_idx('probs', check_type='softmax')

    net.libmodel.startFeatureWriter(
        [data, labels, preds], softmax_idx)
    net.finish_batch()

    if net.multiview_test:
        #  We have to deal with num_samples * num_views
        #  predictions.
        num_views = net.test_data_provider.num_views
        num_samples = labels.shape[1] / num_views
        split_sections = range(
            num_samples, num_samples * num_views, num_samples)
        preds = np.split(preds, split_sections, axis=0)
        labels = np.split(labels, split_sections, axis=1)
        preds = reduce(np.add, preds)
        labels = labels[0]

    return preds, labels
    

class PredictConvNet(convnet.ConvNet):
    _predictions = None
    _option_parser = None

    def make_predictions(self):
        if self._predictions is not None:
            return self._predictions

        num_classes = self.test_data_provider.get_num_classes()
        all_preds = np.zeros((0, num_classes), dtype=np.single)
        all_labels = np.zeros((0, 1), dtype=np.single)
        num_batches = len(self.test_data_provider.batch_range)

        for batch_index in range(num_batches):
            data, labels = self.get_next_batch(train=False)[2]
            preds, labels = make_predictions(self, data, labels, num_classes)
            all_preds = np.vstack([all_preds, preds])
            all_labels = np.vstack([all_labels, labels.T])

        self._predictions = all_preds, all_labels
        return self._predictions

    def write_predictions(self):
        preds, labels = self.make_predictions()
        preds = eval("preds[:, {}]".format(self.op_write_predictions_cols))
        preds = preds.reshape(preds.shape[0], -1)

        print "Predicted true: %.4f" % (
            np.where(preds > preds.max() / 2)[0].shape[0] /
            float(preds.shape[0]))

        with open(self.op_write_predictions_file, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(preds)

    def report(self):
        from sklearn.metrics import auc_score
        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import f1_score
        from sklearn.metrics import precision_recall_curve

        y_pred_probas, y_true = self.make_predictions()
        y_pred = y_pred_probas.argmax(1)
        y_pred_probas = y_pred_probas[:, 1]
        y_true = y_true.reshape(-1)

        print
        print "AUC score:", auc_score(y_true, y_pred_probas)
        print "AUC score (binary):", auc_score(y_true, y_pred)
        print

        print "Classification report:"
        print classification_report(y_true, y_pred)
        print

        print "Confusion matrix:"
        print confusion_matrix(y_true, y_pred)
        print

    def start(self):
        self.op.print_values()
        if self.op_write_predictions_file:
            self.write_predictions()
        if self.op_report:
            self.report()
        sys.exit(0)

    @classmethod
    def get_options_parser(cls):
        if cls._option_parser is not None:
            return cls._option_parser

        op = convnet.ConvNet.get_options_parser()
        op.add_option("write-preds", "op_write_predictions_file",
                      options.StringOptionParser,
                      "Write predictions to this file")
        op.add_option("write-preds-cols", "op_write_predictions_cols",
                      options.StringOptionParser,
                      "Write these prediction columns",
                      default=':')
        op.add_option("report", "op_report",
                      options.BooleanOptionParser,
                      "Do a little reporting?")

        cls._option_parser = op
        return cls._option_parser


def console():
    cfg = sys.argv.pop(1)
    n_sections = len([s for s in get_sections(cfg) if s.startswith('predict')])
    for section in get_sections(cfg):
        if section.startswith('predict'):
            print "=" * len(section)
            print section
            print "=" * len(section)

            # run in a subprocess because of clean-up
            if n_sections > 1:
                p = Process(
                    target=run_model,
                    args=(PredictConvNet, section, cfg),
                    )
                p.start()
                p.join()
            else:
                run_model(PredictConvNet, section, cfg)
