def cap_dataset(batch_size, iters, dataset_size, max_imgs_per_gpu, sample_weight, up_threshold=0.5,
                max_gpu=float("inf")):
    print(f"initial batch_size:\t{batch_size}")
    total = sum(dataset_size.values())
    print(f"-----------------------rounded-----------------------")
    gpus = 1
    batch_size_init = batch_size
    t_wise_batchsize = {}
    while gpus % 8 != 0 or sum(t_wise_batchsize.values()) < batch_size_init:
        t_wise_batchsize = {k: min(v / total * batch_size,
                                   max_imgs_per_gpu[k] * max_gpu
                                   ) for k, v in dataset_size.items()}
        cap = {k: (v == max_imgs_per_gpu[k] * max_gpu) for k, v in t_wise_batchsize.items()}
        t_wise_gpuse = {k: max(1, round(v / max_imgs_per_gpu[k] - up_threshold + 0.5)) for k, v in t_wise_batchsize.items()}
        imgs_per_gpu = {k: round(t_wise_batchsize[k] / v) for k, v in t_wise_gpuse.items()}
        t_wise_batchsize = {k: round(imgs_per_gpu[k] * v) for k, v in t_wise_gpuse.items()}

        t_wise_epochs = {k: iters * t_wise_batchsize[k] / v for k, v in dataset_size.items()}
        batch_size += 1
        gpus = sum(t_wise_gpuse.values())

    loss_weights = {k: round(v * t_wise_batchsize[k], 7) for k, v in sample_weight.items()}
    print(f"{'dataset':15}"
          f"{'batchsize':10}\t"
          f"{'capped':6}\t"
          f"{'per_g_batch':11}(max)\t"
          f"{'epoch':>6}\t"
          f"{'loss_weights':12}\t"
          f"gpus\t"
          f"{'dataset_size':12}\t"
          f"{'sample_weight':13}")
    full_batch = sum(t_wise_batchsize.values())
    for k in t_wise_batchsize.keys():
        print(f"{k:15}{t_wise_batchsize[k]:3}({t_wise_batchsize[k]/full_batch:5.4})\t"
              f"{str(cap[k]):>6}\t"
              f"{imgs_per_gpu[k]:11}({max_imgs_per_gpu[k]:3})\t"
              f"{t_wise_epochs[k]:6.2f}\t"
              f"{loss_weights[k]:12}\t"
              f"{t_wise_gpuse[k]:4}\t"
              f"{dataset_size[k]:12}\t"
              f"{sample_weight[k]:13}")
    print(f"-----------------------------------------------------")
    t_wise_epochs_excap = {k: v for k, v in t_wise_epochs.items() if not cap[k]}
    print(f"{'SUM/AVG(Xcap)':15}{full_batch:9}"
          f"\t{'-':>16}"
          f"\t{'-':>6}"
          f"\t{sum(t_wise_epochs_excap.values())/len(t_wise_epochs_excap):6.2f}"
          f"\t{'-':>12}\t{gpus} ({gpus // 8} nodes)"
          f"\t{sum(dataset_size.values())}")
    print(f"====> iters: {iters}\titers per 100 epochs (excluding capped dataset): {round(100/(sum(t_wise_epochs_excap.values())/len(t_wise_epochs_excap))*iters)}")

reid_w=2
pose_w=2000
parse_w=1
attr_w=.001
cap_dataset(batch_size=3200,
            iters=171335,
            dataset_size={'reid':118_063,   'dgmarket':128_309,
                          #pose
                          'pose': 149_813, 'aic': 378_352,  'posetrack': 97_174,
                          'mhp': 40_437, '3dpw': 68_663, 'penn': 34_475, 'halpe': 41_263, 'h36m': 312_187,
                          #parse
                          'h36_parse': 62_668, 'LIP': 30_462, 'CIHP': 28_280, 'deepfasion': 191_961,
                          #attr
                          'pa100k':90_000, 'rapv2': 67_943,
                          'hardhc': 28_336, 'uavhuman': 16_183, 'parse27k': 27_482, 'duke': 34_183, 'market': 12_936,
                          #det
                          'CrowdHuman': 15_000,
                         },
            max_imgs_per_gpu={'reid':400, 'dgmarket':400,
                              #pose
                              'pose': 168, 'aic':174, 'posetrack':168,
                              'mhp': 168, '3dpw': 168, 'penn': 168, 'halpe': 168, 'h36m': 168,
                              #parse
                              'h36_parse': 26, 'LIP': 29, 'CIHP': 29,'deepfasion': 30,
                              #attr
                              'pa100k':230,'rapv2': 230,
                              'hardhc': 230, 'uavhuman': 230, 'parse27k': 230, 'duke': 230, 'market': 230,
                              #det
                              'CrowdHuman': 1,
                             },
            sample_weight={'reid': reid_w,'dgmarket': reid_w,
                           #pose
                           'pose':pose_w,  'aic':pose_w,   'posetrack':pose_w,
                           'mhp': pose_w, '3dpw': pose_w, 'penn': pose_w, 'halpe': pose_w, 'h36m': pose_w,
                           #parse
                           'h36_parse': parse_w, 'LIP': parse_w, 'CIHP': parse_w, 'deepfasion': parse_w,
                           #attr
                           'pa100k':attr_w,'rapv2': attr_w,
                           'hardhc':attr_w, 'uavhuman':attr_w, 'parse27k':attr_w, 'duke':attr_w, 'market':attr_w,
                           #det
                           'CrowdHuman': 10,
                          },
            up_threshold=0.15,
            max_gpu=20
           )



# example usage
# reid_w=2
# pose_w=2000
# parse_w=1
# attr_w=.001
# cap_dataset(batch_size=3200,
#             iters=11250,
#             dataset_size={'reid':118_063,   'dgmarket':128_309,
#                           #pose
#                           'pose': 149_813, 'aic': 378_352,  'posetrack': 97_174,
#                           'mhp': 40_437, '3dpw': 68_663, 'penn': 34_475, 'halpe': 41_263, 'h36m': 312_187,
#                           #parse
#                           'h36_parse': 62_668, 'LIP': 30_462, 'CIHP': 28_280, 'deepfasion': 191_961,
#                           #attr
#                           'pa100k':90_000, 'rapv2': 67_943,
#                           'hardhc': 28_336, 'uavhuman': 16_183, 'parse27k': 27_482, 'duke': 34_183, 'market': 12_936,
#                          },
#             max_imgs_per_gpu={'reid':400, 'dgmarket':400,
#                               #pose
#                               'pose': 168, 'aic':174, 'posetrack':168,
#                               'mhp': 168, '3dpw': 168, 'penn': 168, 'halpe': 168, 'h36m': 168,
#                               #parse
#                               'h36_parse': 26, 'LIP': 29, 'CIHP': 29,'deepfasion': 30,
#                               #attr
#                               'pa100k':210,'rapv2': 210,
#                               'hardhc': 210, 'uavhuman': 210, 'parse27k': 210, 'duke': 210, 'market': 210,},
#             sample_weight={'reid': reid_w,'dgmarket': reid_w,
#                            #pose
#                            'pose':pose_w,  'aic':pose_w,   'posetrack':pose_w,
#                            'mhp': pose_w, '3dpw': pose_w, 'penn': pose_w, 'halpe': pose_w, 'h36m': pose_w,
#                            #parse
#                            'h36_parse': parse_w, 'LIP': parse_w, 'CIHP': parse_w, 'deepfasion': parse_w,
#                            #attr
#                            'pa100k':attr_w,'rapv2': attr_w,
#                            'hardhc':attr_w, 'uavhuman':attr_w, 'parse27k':attr_w, 'duke':attr_w, 'market':attr_w,},
#             up_threshold=0.15
#            )
# output:
# initial batch_size:	3200
# -----------------------rounded-----------------------
# dataset      batchsize	per_g_batch(max)	 epoch	loss_weights	gpus	dataset_size	sample_weight
# reid               235	        235(400)	 22.39	         470	   1	      118063	            2
# dgmarket           255	        255(400)	 22.36	         510	   1	      128309	            2
# pose               298	        149(168)	 22.38	      596000	   2	      149813	         2000
# aic                750	        150(174)	 22.30	     1500000	   5	      378352	         2000
# posetrack          194	         97(168)	 22.46	      388000	   2	       97174	         2000
# mhp                 80	         80(168)	 22.26	      160000	   1	       40437	         2000
# 3dpw               137	        137(168)	 22.45	      274000	   1	       68663	         2000
# penn                69	         69(168)	 22.52	      138000	   1	       34475	         2000
# halpe               82	         82(168)	 22.36	      164000	   1	       41263	         2000
# h36m               620	        155(168)	 22.34	     1240000	   4	      312187	         2000
# h36_parse          125	         25( 26)	 22.44	         125	   5	       62668	            1
# LIP                 60	         30( 29)	 22.16	          60	   2	       30462	            1
# CIHP                56	         28( 29)	 22.28	          56	   2	       28280	            1
# deepfasion         377	         29( 30)	 22.09	         377	  13	      191961	            1
# pa100k             179	        179(210)	 22.38	       0.179	   1	       90000	        0.001
# rapv2              135	        135(210)	 22.35	       0.135	   1	       67943	        0.001
# hardhc              56	         56(210)	 22.23	       0.056	   1	       28336	        0.001
# uavhuman            32	         32(210)	 22.25	       0.032	   1	       16183	        0.001
# parse27k            55	         55(210)	 22.51	       0.055	   1	       27482	        0.001
# duke                68	         68(210)	 22.38	       0.068	   1	       34183	        0.001
# market              26	         26(210)	 22.61	       0.026	   1	       12936	        0.001
# -----------------------------------------------------
# SUM/AVG           3889	               -	 22.36	           -	48 (6 nodes)
# ====> iters: 11250	iters per 100 epochs: 50321