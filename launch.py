if __name__ == '__main__':

    import logging
    import matplotlib.pyplot as plt

    from timeit import default_timer as timer

    from mosartwmpy.mosartwmpy import Model

    # launch simulation
    mosart_wm = Model()
    mosart_wm.initialize('./config.yaml')
    t = timer()
    mosart_wm.update_until(mosart_wm.get_end_time())
    logging.info(f'Simulation completed in {mosart_wm.pretty_timer(timer() - t)}.')
