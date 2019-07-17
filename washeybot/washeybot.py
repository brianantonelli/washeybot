#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import locale
locale.setlocale(locale.LC_ALL, 'C')  # needed for tesserocr

from tesserocr import PyTessBaseAPI, PSM, OEM
