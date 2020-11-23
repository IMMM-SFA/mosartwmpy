import warnings
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from src.mosart import Mosart
warnings.filterwarnings('ignore')
if __name__ == '__main__':
    dask_cluster = SLURMCluster(
        queue='short',
        walltime='01:00:00',
        project="IM3",
        cores=24,
        processes=1,
        memory="64GB",
    )
    dask_cluster.scale(jobs=1)
    dask_client = Client(dask_cluster)
    self = Mosart()
    self.initialize()
    self.update_until(self.get_end_time())
    
    self.grid.to_parquet(
        f'./output/{self.name}/grid',
        overwrite=True
    )
    self.state.to_parquet(
        f'./output/{self.name}/state',
        overwrite=True
    )