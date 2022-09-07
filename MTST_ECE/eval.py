from MTST_ECE.utils.Metrics import emotion_metric, cause_metric, pair_metric
from MTST_ECE.utils.PrepareData import convert_document_to_ids, ExtractPairs

# import mindspore
# from Utils.tokenization import FullTokenizer, convert_tokens_to_ids
# from Utils.Metrics import emotion_metric, cause_metric, pair_metric
# from sklearn.metrics import precision_score, recall_score, f1_score




# import torch
# from Config import Config

# config = Config()
#
# tokenizer = FullTokenizer(config.bert_path + "vocab.txt")
#
# def padding_and_mask(ids_list):
#     max_len = max([len(x) for x in ids_list])
#     mask_list = []
#     ids_padding_list = []
#     for ids in ids_list:
#         mask = [1.] * len(ids) + [0.] * (max_len - len(ids))
#         ids = ids + [0] * (max_len - len(ids))
#         mask_list.append(mask)
#         ids_padding_list.append(ids)
#     return ids_padding_list, mask_list
#
# def convert_document_to_ids(document_list):
#
#     text_list, tokens_list, ids_list = [], [], []
#     ## The clauses in each document are splited by '\x01'
#     document_len = [len(x.split('\x01')) for x in document_list]
#
#     for document in document_list:
#         text_list.extend(document.strip().split('\x01'))
#     for text in text_list:
#         text = ''.join(text.split())
#         tokens = tokenizer.tokenize(text)
#         tokens = ["[CLS]"] + tokens + ["[SEP]"]
#         tokens_list.append(tokens)
#     for tokens in tokens_list:
#         ids_list.append(convert_tokens_to_ids(config.bert_path + "vocab.txt", tokens))
#
#     ids_padding_list, mask_list = padding_and_mask(ids_list)
#     ids_padding_tensor = mindspore.Tensor(ids_padding_list, mindspore.int64)
#     mask_tensor = mindspore.Tensor(mask_list)
#     return ids_padding_tensor, mask_tensor, document_len


# def ExtractPairs(tag_pred, len_list, config):
#     start, scope = 0, config.scope
#     ext_pairs = []
#     for dl in len_list:
#         end = start + dl
#         doc_tag = tag_pred[start: end]
#         pair = []
#         for inx, tag in enumerate(doc_tag):
#             if -scope <= tag-scope <= scope:
#                 pair.append((inx+tag-scope, inx))
#         ext_pairs.append(pair)
#         start = end
# #     print (tag_pred)
#     return ext_pairs
#
# def pair2e_c_label(documents_len, data):
#     emo_grounds, cau_grounds = [], []
#     pair_list = data[1]
#     for i, dl in enumerate(documents_len):
#         pair = pair_list[i]
#         emotion = [0] * dl
#         cause = [0] * dl
#         for p in pair:
#             emotion[p[0]] = 1
#             cause[p[1]] = 1
#         emo_grounds.extend(emotion)
#         cau_grounds.extend(cause)
#
#     return emo_grounds, cau_grounds

class EvalEngine:
    def __init__(self, model):
        self.model = model
        self.model.set_train(False)

    def eval(self, data, batch_size, scope):
        data_len = len(data[0])
        documents_len = [len(x.split('\x01')) for x in data[0]]
        pair_grounds = data[1]
        batch_i = 0
        emo_preds, cau_preds, pair_preds = [], [], []
        while batch_i * batch_size < data_len:
            start, end = batch_i * batch_size, (batch_i + 1) * batch_size
            document_list = data[0][start: end]
            len_list = documents_len[start: end]
            ids_padding_tensor, mask_tensor, document_len = convert_document_to_ids(document_list)
            retag_probs, emo_probs, cau_probs = self.model(ids_padding_tensor, mask_tensor, document_len, None, 0)
            tag_pred = retag_probs.argmax(1).asnumpy().tolist()
            emo_pred = emo_probs.argmax(1).asnumpy().tolist()
            cau_pred = cau_probs.argmax(1).asnumpy().tolist()
            pair_pred = ExtractPairs(tag_pred, len_list, scope)
            emo_preds.extend(emo_pred)
            cau_preds.extend(cau_pred)
            pair_preds.extend(pair_pred)
            batch_i += 1
            print(f'eval/test batch: {batch_i}')
        emo_metric = emotion_metric(pair_preds, pair_grounds)
        cau_metric = cause_metric(pair_preds, pair_grounds)
        pr_metric = pair_metric(pair_preds, pair_grounds)
        return emo_metric, cau_metric, pr_metric


# def Performance(net, data, config):
#     data_len = len(data[0])
#     documents_len = [len(x.split('\x01')) for x in data[0]]
#     # emo_grounds, cau_grounds = pair2e_c_label(documents_len, data)
#     pair_grounds = data[1]
#     batch_i = 0
#     net.set_train(False)
#     #base_encoder.eval()
#     #sl_model.eval()
#     emo_preds, cau_preds, pair_preds = [], [], []
#     while batch_i * config.batch_size < data_len:
#         # if batch_i < 60:
#         #     continue
#         # elif batch_i == 60:
#         #     pass
#         # else:
#         #     break
#         start, end = batch_i * config.batch_size, (batch_i +1) * config.batch_size
#         document_list = data[0][start: end]
#         # pair_true_list = pair_grounds[start: end]
#         len_list = documents_len[start: end]
#         ids_padding_tensor, mask_tensor, document_len = convert_document_to_ids(document_list)
#         retag_probs, emo_probs, cau_probs = net(ids_padding_tensor, mask_tensor, document_len, None, 0)
#         #_, clause_state_list = base_encoder(document_list)
#         #retag_probs, emo_probs, cau_probs = sl_model(clause_state_list, None, 'eval')
#         tag_pred = retag_probs.argmax(1).asnumpy().tolist()
#         emo_pred = emo_probs.argmax(1).asnumpy().tolist()
#         cau_pred = cau_probs.argmax(1).asnumpy().tolist()
#         pair_pred = ExtractPairs(tag_pred, len_list, config)
#
#         emo_preds.extend(emo_pred)
#         cau_preds.extend(cau_pred)
#         pair_preds.extend(pair_pred)
#         batch_i += 1
#
#     emo_metric = emotion_metric(pair_preds, pair_grounds)
#     cau_metric = cause_metric(pair_preds, pair_grounds)
#     pr_metric = pair_metric(pair_preds, pair_grounds)
#
#     return emo_metric, cau_metric, pr_metric

'''def Performance(base_encoder, sl_model, data, config):   
    data_len = len(data[0])
    documents_len = [len(x.split('\x01')) for x in data[0]]
    emo_grounds, cau_grounds = pair2e_c_label(documents_len, data)
    pair_grounds = data[1]
    batch_i = 0
    base_encoder.eval()
    sl_model.eval()
    emo_preds, cau_preds, pair_preds = [], [], []
    while batch_i * config.batch_size < data_len:
        start, end = batch_i * config.batch_size, (batch_i +1) * config.batch_size
        document_list = data[0][start: end]
        pair_true_list = pair_grounds[start: end]
        len_list = documents_len[start: end]
        _, clause_state_list = base_encoder(document_list)
        retag_probs, emo_probs, cau_probs = sl_model(clause_state_list, None, 'eval')
        tag_pred = retag_probs.argmax(1).data.cpu().numpy().tolist()
        emo_pred = emo_probs.argmax(1).data.cpu().numpy().tolist()
        cau_pred = cau_probs.argmax(1).data.cpu().numpy().tolist()
        pair_pred = ExtractPairs(tag_pred, len_list, config)
        
        emo_preds.extend(emo_pred)
        cau_preds.extend(cau_pred)
        pair_preds.extend(pair_pred)
        batch_i += 1
    
    emo_metric = emotion_metric(pair_preds, pair_grounds)
    cau_metric = cause_metric(pair_preds, pair_grounds)
    pr_metric = pair_metric(pair_preds, pair_grounds)
    
    return (emo_metric, cau_metric, pr_metric)'''
