    #---------------------->  att avaluate   <-------------------------------#
        #     output = outputs[0]
        #     att_pre = output[:,7:10]
        #     att_pre = torch.sort(att_pre, dim=1)   
        #     num_detection = targets.shape[1]
        #     tol = tol + num_detection

        #     att_bbox = output[:, 0:4]
 
        #     match_id = 0
        #     match = 10000
        #     # print(targets.shape)   #(b,num,14)
        #     # print(targets)
        #     for att_id in range(self.num_atts):
        #         for num_id in range(num_detection):
                    
        #             if targets[0, num_id, 5 + att_id]==1:
        #                 pos_tol[att_id] = pos_tol[att_id] + 1 
        #                 for bb in range(output.shape[0]):
        #                     abs_match = torch.sum(torch.abs(output[bb, 0:4] - targets[0, num_id, 0:4].cuda()))
        #                     # print(abs_match)
        #                     if abs_match < match:
        #                         match_id = bb
        #                         match = abs_match
    
        #                 for i in range(3):
        #                     if output[match_id, 7+i] == att_id and (match_id< num_id or match_id==num_id):
        #                         pos_cnt[att_id] = pos_cnt[att_id] + 1
        #                 match = 10000
                    
        #             if targets[0, num_id, 5 + att_id]==0:
        #                 neg_tol[att_id] = neg_tol[att_id] + 1 

        #                 for bb in range(output.shape[0]):
        #                     abs_match = torch.sum(torch.abs(output[bb, 0:4] - targets[0, num_id, 0:4].cuda()))
        #                     # print(abs_match)
        #                     if abs_match < match:
        #                         match_id = bb
        #                         match = abs_match
        #                 for i in range(3):
        #                     if output[match_id, 7+i] != att_id and (match_id< num_id or match_id==num_id) and i == 2:
        #                         neg_cnt[att_id] = neg_cnt[att_id] + 1  
        #                 match =  10000
        #     print("pos_tol",pos_tol)  
        #     print("pos_cnt", pos_cnt)   
        #     match=10000
        #     for num_id in range(num_detection):
        #         tp = 0
        #         fn = 0
        #         fp = 0
        #         for att_id in range(self.num_atts):
        #             for bb in range(output.shape[0]):
        #                 abs_match = torch.sum(torch.abs(output[bb, 0:4] - targets[0, num_id, 0:4].cuda()))
        #                 if abs_match < match:
        #                     match_id = bb
        #                     match = abs_match
        #             for i in range(3):
        #                 if output[match_id, 7+i]==att_id and targets[0, num_id, 5 + att_id] == 1:
        #                     tp = tp + 1
        #                 elif output[match_id, 7+i]==att_id and targets[0, num_id, 5 + att_id] == 0:
        #                     fp = fp +1
        #             for i in range(3):
        #                 if output[match_id, 7+i] != att_id and targets[0, num_id, 5 + att_id] == 1 and i==2:
        #                     fn = fn + 1
        #             match = 10000
                
        #         if tp + fn + fp != 0:
        #             accu = accu +  1.0 * tp / (tp + fn + fp)
        #         if tp + fp != 0:
        #             prec = prec + 1.0 * tp / (tp + fp)
        #         if tp + fn != 0:
        #             recall = recall + 1.0 * tp / (tp + fn)

        # print('=' * 100)
        # print('\t     Attr              \tp_true/n_true\tp_tol/n_tol\tp_pred/n_pred\tcur_mA')
        # mA = 0.0
        # for it in range(self.num_atts):
        #     cur_mA = ((1.0*pos_cnt[it]/pos_tol[it]) + (1.0*neg_cnt[it]/neg_tol[it])) / 2.0
        #     mA = mA + cur_mA
        #     print('\t#{:2}: {:4}\{:4}\t{:4}\{:4}\t{:4}\{:4}\t{:.5f}'.format(it,pos_cnt[it],neg_cnt[it],pos_tol[it],neg_tol[it],(pos_cnt[it]+neg_tol[it]-neg_cnt[it]),(neg_cnt[it]+pos_tol[it]-pos_cnt[it]),cur_mA))
        # mA = mA / self.num_atts
        # print('\t' + 'mA:        '+str(mA))

       
        # accu = accu / tol
        # prec = prec / tol
        # recall = recall / tol
        # f1 = 2.0 * prec * recall / (prec + recall)
        # print('\t' + 'Accuracy:  '+str(accu))
        # print('\t' + 'Precision: '+str(prec))
        # print('\t' + 'Recall:    '+str(recall))
        # print('\t' + 'F1_Score:  '+str(f1))
        # print('=' * 100)