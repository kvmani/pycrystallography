'''
Created on 18-Jan-2018

@author: Admin
'''
import pstats
p = pstats.Stats('restats')
p.sort_stats('cumulative').print_stats(10)
# p.sort_stats('time').print_stats(50)
