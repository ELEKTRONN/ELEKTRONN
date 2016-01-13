# -*- coding: utf-8 -*-
# ELEKTRONN - Neural Network Toolkit
#
# Copyright (c) 2014 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Marius Killinger, Gregor Urban


def pprinttime(t):
    """Custom printing of elapsed time"""
    if t > 4000:
        s = 't=%.1fh' % (t / 3600)
    elif t > 300:
        s = 't=%.0fm' % (t / 60)
    else:
        s = 't=%.0fs' % (t)
    return s
