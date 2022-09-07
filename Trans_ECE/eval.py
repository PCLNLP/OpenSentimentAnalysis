from Trans_ECE.utils.Metrics import emotion_metric, cause_metric, pair_metric
from Trans_ECE.utils.PrepareData import convert_document_to_ids, merge_tuple_single


class EvalEngine:
    def __init__(self, model):
        self.model = model
        self.model.set_train(False)

    def eval(self, data, batch_size):
        data_len = len(data[0])
        documents_len = [len(x.split('\x01')) for x in data[0]]
        grounds = data[1]
        batch_i = 0
        single_predicts, tuple_predicts = [], []
        while batch_i * batch_size < data_len:
            start, end = batch_i * batch_size, (batch_i +1) * batch_size
            document_list = data[0][start: end]
            ids_padding_tensor, mask_tensor, document_len = convert_document_to_ids(document_list)
            single_preds, tuple_preds = self.model(ids_padding_tensor, mask_tensor, document_len, None, None, 0)
            single_predicts.extend(single_preds)
            tuple_predicts.extend(tuple_preds)
            batch_i += 1
            print(f"eval/test batch: {batch_i}")
        final_preds = merge_tuple_single(single_predicts, tuple_predicts, data[0])
        emo_metric = emotion_metric(final_preds, grounds, documents_len)
        cse_metric = cause_metric(final_preds, grounds, documents_len)
        pr_metric = pair_metric(final_preds, grounds)
        return emo_metric, cse_metric, pr_metric
