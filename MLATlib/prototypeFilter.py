# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 21:54:21 2020

@author: Till
"""



x_nodes = aco.x_cart_nonan[aco.idsn_np]

cos_score = np.ones(len(x_nodes))
Diff = np.diff(x_nodes, axis=0)
cos_score[1:-1] = np.sum(Diff[0:-1] * (Diff[1:]), axis=1)\
    / (la.norm(Diff[0:-1], axis=1)\
       * la.norm(Diff[1:], axis=1)
       )
cos_score[0] = cos_score[1]
cos_score[-1] = cos_score[-2]

seg_score = (cos_score[0:-1] + cos_score[1:]) / 2

scores = [[s] * len(
    aco.ids[(aco.ids >= idsn[i]) & (aco.ids < idsn[i+1])]
    )
    for i, s in enumerate(seg_score)
    ]
scores[-1].append(seg_score[-1])  # Off by 1 fix

flat_scores = [item for sublist in scores for item in sublist]


# Flag the curvature score
aco.TRAac.loc[~np.isnan(aco.TRAac['lat']), 'score'] = flat_scores

