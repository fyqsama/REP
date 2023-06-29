NA_PRIMITIVES = [
  'sage',
  'sage_sum',
  'sage_max',
  'gcn',
  'gin',
  'gat',
  'gat_sym',
  'gat_cos',
  'gat_linear',
  'gat_generalized_linear',
  'geniepath',
]


SC_PRIMITIVES=[
  'none',
  'skip',
]


LA_PRIMITIVES=[
  'l_max',
  'l_concat',
  'l_lstm'
]

SANE = 'gat_generalized_linear||gat||gat_sym||skip||skip||skip||l_max'
REP = 'gat_cos||gin||gat_generalized_linear||skip||skip||skip||l_concat'
