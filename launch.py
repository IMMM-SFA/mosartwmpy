if __name__ == '__main__':

    import logging
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    from mosartwmpy.mosartwmpy import Model

    # launch simulation
    mosart_wm = Model()
    mosart_wm.initialize('./config.yaml')
    mosart_wm.update_until(mosart_wm.get_end_time())
