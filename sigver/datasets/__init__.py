from .gpds import GPDSDataset
from .gpds_synth import GPDSSynthDataset
from .mcyt import MCYTDataset
from .cedar import CedarDataset
from .brazilian import BrazilianDataset, BrazilianDatasetWithoutSimpleForgeries

available_datasets = {'gpds': GPDSDataset,
                      'gpds_synth': GPDSSynthDataset,
                      'mcyt': MCYTDataset,
                      'cedar': CedarDataset,
                      'brazilian': BrazilianDataset,
                      'brazilian-nosimple': BrazilianDatasetWithoutSimpleForgeries}
