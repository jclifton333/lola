import numpy as np


def util_welfare(vi, vmi):
  return vi + vmi


def selfish_welfare(vi, vmi):
  return vi


def welfare_factory(welfare_name):
  if welfare_name == 'self':
    return selfish_welfare
  elif welfare_name == 'util':
    return util_welfare


