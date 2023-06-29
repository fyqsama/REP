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

arch0 = 'gat_cos||gin||gat_linear||skip||skip||skip||l_max'
arch1 = 'gat_cos||gin||gat_linear||skip||skip||skip||l_concat'
arch2 = 'gat_cos||gin||gat_generalized_linear||skip||skip||skip||l_concat'
arch3 = 'sage||gin||gat_generalized_linear||skip||skip||skip||l_concat'
arch4 = 'sage||gin||gin||skip||skip||skip||l_concat'
arch5 = 'sage||gin||gin||skip||skip||skip||l_max'
arch6 = 'gat_linear||gin||gin||skip||skip||skip||l_max'
arch7 = 'gat_linear||gin||gat_linear||skip||skip||skip||l_max'
arch8 = 'gat_linear||sage_max||gat_linear||skip||skip||skip||l_max'
arch9 = 'gat_linear||sage_max||sage_sum||skip||skip||skip||l_max'
arch10 = 'gat_linear||sage_sum||sage_sum||skip||skip||skip||l_max'
arch11 = 'sage_sum||sage_sum||sage_sum||skip||skip||skip||l_max'
arch12 = 'sage_sum||sage_sum||sage_max||skip||skip||skip||l_max'
