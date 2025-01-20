# -*- coding: utf-8 -*-
"""Cp2kMobilityWorkChain workchain"""

from argparse import Namespace
from cProfile import label
import io
import ase
import pymatgen
import os, time

import numpy as np
from ase.atoms import Atoms
import ruamel.yaml as yaml
from copy import deepcopy
from scipy.stats import linregress
import click
from aiida.common import AttributeDict, NotExistent
from aiida.engine import ToContext, WorkChain, if_, calcfunction, while_
from aiida.orm import Code, Dict, SinglefileData, StructureData, Str, Float, RemoteData
from aiida.plugins import CalculationFactory, DataFactory, WorkflowFactory
from aiida_lsmo.utils import get_structure_from_cif, aiida_dict_merge, HARTREE2EV
from aiida.orm import QueryBuilder, Node, load_node
from aiida_lsmo.utils.multiply_unitcell import check_resize_unit_cell_legacy
from aiida_lsmo.utils.cp2k_utils_master import get_kinds_section
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.elasticity.strain import DeformedStructureSet

from pymatgen.analysis.elasticity import Strain, Deformation
from pymatgen.analysis.elasticity import Stress
from pymatgen.analysis.elasticity import ElasticTensor, ElasticTensorExpansion,  NthOrderElasticTensor, diff_fit, find_eq_stress

import sumo
from sumo.electronic_structure.effective_mass import get_fitting_data, fit_effective_mass
from aiida.tools import get_kpoints_path
from pymatgen.electronic_structure.core import Spin, Orbital
from pymatgen.electronic_structure.bandstructure import BandStructure, Kpoint, BandStructureSymmLine

from mof_vac_level import MOFVacLevel

#Constants for equation 

import math
QCHARGE_C =  1.602176634E-19
GPA_TO_PA = 1E9
MASSELECTRON_KG = 9.1093837015E-31
KB_T_J = 4.141947E-21
H_BAR_JS = 6.62607015E-34
CONSTANT = math.pow((H_BAR_JS/(2*math.pi)),4) * math.sqrt(8*math.pi)/3
EV2J = 1.60218E-19
VAL_ELEC = {'Ag': 19, 'Al': 3, 'Ar': 8, 'As': 5, 'B': 3, 'Be': 4, 'Br': 7, 'C': 4, 'Ca': 10, 'Cd': 20, 'Cl': 7, 'Co': 17, 'Cr': 14, 'Cu': 19,
 'F': 7, 'Fe': 16, 'Ga': 3, 'H': 1, 'He': 2, 'I': 7, 'In': 3, 'K': 9, 'Li': 3, 'Mg': 2, 'Mn': 15, 'N': 5, 'Na': 9, 'Ne': 8, 'Ni': 18, 'O': 6,
 'P': 5, 'S': 6, 'Sc': 11, 'Si': 4, 'Ti': 12, 'V': 13, 'Zn': 12, 'Zr': 12, 'Ru': 16}

# import sub-workchains
Cp2kBaseWorkChain = WorkflowFactory('cp2k.base')
Cp2kMultistageWorkChain = WorkflowFactory('lsmo.cp2k_multistage')
Cp2kBandsWorkChain = WorkflowFactory('photocat_workchains.bandstructure')
THIS_DIR = os.path.dirname(os.path.realpath(__file__))



#import aiida data
CifData = DataFactory('cif')  # pylint: disable=invalid-name
def merge_dict(dct, merge_dct):
    """ Taken from https://gist.github.com/angstwad/bf22d1822c38a92ec0a9
    Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, merge_dict recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct (overwrites dct data if in both)
    :return: None
    """
    import collections  # pylint:disable=import-outside-toplevel
    for k, _ in merge_dct.items():  # it was .iteritems() in python2
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            merge_dict(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


def last_stressfilepath(path):
    """ 
        Function to located the latest stress_stensor, since cp2k version ADD_LAST functionality not workin

        Arguments:
        path: Absolue path in the repository where the stress tensor files are located
        
        Returns:
        (string) The path + the name of the latested stress_tensor from the Def GEO calculations

        
    """
    mypathlist = []
    for file in os.listdir(path):
        if file.endswith(".stress_tensor"):
            mypathlist.append(file)

    last_stress = [int(mypaths[15:-14]) for mypaths in mypathlist]
    index_last_stress = last_stress.index(max(last_stress))
    the_last_path = os.path.join(path, mypathlist[index_last_stress])
    return the_last_path

def run_seekpath(structure):
    parameters = {
            'symprec': 0.05,
            'angle_tolerance': -1.0
        }
    seekpath_result = get_kpoints_path(structure, **parameters)
    return seekpath_result['primitive_structure']

def get_info_fromBands(aiidaout_path):
    """
    Function to retrieved the fermi , and homo and lumo orbital number

    """
    my_info = {}
    with open(aiidaout_path, 'r') as my_out:
        for line in my_out.readlines():
            if 'Fermi Energy' in line:
                my_info['fermi'] = float(line.split()[-1])
            if 'Number of occupied orbitals' in line:
                my_info['homo'] = int(line.split()[-1])
                my_info['lumo'] = int(line.split()[-1]) + 1
    
    return my_info

def get_pmg_bands(bands_data, structure, fermi):
    """ 
        Function to generate a BandStructureSymmLine object to be used by sumo

        Arguments:
        bands_data : BandsData from aiida node
        structure =  Aiida structure 
        fermi: fermi energy level 
        
        Returns:
        (BandStructureSymmLine) 

        
    """
    structure_data_primitive = run_seekpath(structure) #Use the primitive 
    pymatgen_structure = structure_data_primitive.get_pymatgen_structure()

    my_kpoints = bands_data.get_kpoints()
    my_labels = bands_data.labels
    label_dict = {label: my_kpoints[index] for index,label in my_labels}

    bs_pmg = BandStructureSymmLine(kpoints=my_kpoints, 
                                   eigenvals={Spin.up: bands_data.get_bands().T},
                                   lattice=pymatgen_structure.lattice.reciprocal_lattice, 
                                   efermi=fermi, 
                                   labels_dict=label_dict,
                                   structure=pymatgen_structure)
    
    return bs_pmg

def call_sumo(bs, carrier='hole'):
    """
    Arguments:
        bs: Pymatgen BandStructureSymmLine object
        carrier
    Returns:
        (float) mass of lightest band of that type
    """
    masses = []
    if carrier == 'hole': 
        extreme = bs.get_vbm()
    elif carrier == 'electron': 
        extreme = bs.get_cbm()
        
    for spin in [Spin.up, Spin.down]:
        if carrier == 'electron':
            b_ind = list(extreme['band_index'][spin])
            if len(b_ind) != 0:
                b_ind_num = min(b_ind)
            else:
                break
        elif carrier == 'hole':
            b_ind = list(extreme['band_index'][spin])
            if len(b_ind) != 0:
                b_ind_num = max(b_ind)
            else:
                break

        for i in extreme['kpoint_index']:
            fit_data = get_fitting_data(bs, spin, 
                b_ind_num, i, 4)[0]
            masses.append(fit_effective_mass(fit_data['distances'],
                fit_data['energies']))
        
    return min([abs(mass) for mass in masses]) #Only returns the min 

def add_condband(structure):
    """Add 30% of conduction bands to the CP2K input. If 30 % is 0, then add only one."""
    total = 0
    for symbol in structure.get_ase().get_chemical_symbols():
        total += VAL_ELEC[symbol]
    added_mos = total // 30  # 30% of conduction band
    if added_mos == 0:
        added_mos = 1
    return added_mos

@calcfunction
def structure_with_pbc(s):
    atoms = s.get_ase()
    atoms.pbc = True
    new_s = StructureData(ase=atoms)
    return new_s

class Cp2kMobilityWorkChain(WorkChain):
    """A workchain to compute Bardeen and Schocklet mobility using CP2K , inspired by the work of Muschielok et at (2019)(J. Chem. Phys. 151, 015102 (2019); doi: 10.1063/1.5108995) 
    This class combines MultistageWorkChain + Cp2kBandsWorkChain + Cp2kBandsWorkChain to determine the effective mass, bulk modulus, and deformation potential.
    """

    @classmethod
    def define(cls, spec):
        """Specify input, outputs, and the workchain outline."""
        super().define(spec)

        spec.input('structure', valid_type=CifData, help='Initial structure be optimized')
        spec.input('protocol_geocell', valid_type=Str, default=lambda: Str('geocell'), help='Yaml file,  DFT parameters, GEO and CELL stages') ## Geocell pparameters
        spec.input('protocol_geo', valid_type=Str, default=lambda: Str('geo'), help='Yaml file,  DFT parameters , kpoint GEO') ## Geocell pparameters
        spec.input('protocol_bands', valid_type=Str, default=lambda: Str('bands'), help='Yaml file,  DFT parameters, Bandstructure') ## Geocell pparameters
        spec.expose_inputs(Cp2kMultistageWorkChain, namespace='cp2k_multistage', exclude=['structure']) # 
        spec.expose_inputs(Cp2kBaseWorkChain, namespace='cp2k_def_geo', exclude=['cp2k.structure', 'cp2k.parameters'])
        spec.expose_inputs(Cp2kBandsWorkChain, namespace='cp2k_def_bands', exclude=['structure'])
        spec.outline(
            cls.setup,
            cls.run_geocell,
            cls.generate_deformations,
            cls.geoopt_kpoints,
            cls.band_structure,
            cls.run_geoopt_deformations,
            cls.run_bandsgeo_def,
            cls.run_bands_deformation,
            cls.vacuum_pore,
            cls.results
        )
        spec.expose_outputs(Cp2kMultistageWorkChain)
        spec.expose_outputs(Cp2kBaseWorkChain)
        spec.expose_outputs(Cp2kBandsWorkChain)
        spec.output('out_data',
                    valid_type=Dict,
                    help='Dictionary with the energies, deformations, and tensor')


    def setup(self):
        """Perform initial setup."""
        self.ctx.structure = self.inputs.structure
        if 'protocol_geocell' in self.inputs:
            self.ctx.protocol_geocell = self.inputs.protocol_geocell.value 
        else:
            self.ctx.protocol_geocell = os.path.abspath(os.path.join(THIS_DIR, 'mobility_protocols/geocell_2.yaml'))

        if 'protocol_bands' in self.inputs:
            self.ctx.protocol_bands = self.inputs.protocol_bands.value

        else:
            self.ctx.protocol_bands = os.path.join(THIS_DIR, 'mobility_protocols/bands_2.yaml')

        if 'protocol_geo' in self.inputs:
            self.ctx.protocol_geo = self.inputs.protocol_geo.value

        else:
            self.ctx.protocol_geo = os.path.abspath(os.path.join(THIS_DIR, 'mobility_protocols/geo_2.yaml'))
        #self.ctx.protocol_geo = 
        #self.ctx.protocol_bands = 



    def run_geocell(self):
        #Run a GEO/CELL 
        ms_inputs = AttributeDict(self.exposed_inputs(Cp2kMultistageWorkChain, namespace='cp2k_multistage'))
        ms_inputs['structure'] = get_structure_from_cif(self.inputs.structure)
        ms_inputs['protocol_yaml'] = SinglefileData(file=self.ctx.protocol_geocell) #Protocol of GEO/CELL
        ms_inputs['cp2k_base']['cp2k']['settings'] = Dict(dict={'additional_retrieve_list': ['*.stress_tensor']}) #Retrieve stress_tensor files to repository
        running = self.submit(Cp2kMultistageWorkChain, **ms_inputs)
        return ToContext(ms_wc=running)

    def generate_deformations(self):
        """Generate the deformations for the calculation of the bulk modulus"""
        initial_struc = structure_with_pbc(self.ctx.ms_wc.outputs.output_structure) #Use optimized structure
        my_pymatgen_struc = initial_struc.get_pymatgen()   
        #Define the strains to apply 
        my_norm_strains=[-0.02,-0.01,0.01, 0.02]
        my_shear_strains=[-0.02,-0.01,0.01, 0.02]
        # DeformedStructureSet object containing the strains, and unstrained pymatgen Structure
        self.ctx.def_dict = DeformedStructureSet(my_pymatgen_struc, norm_strains=my_norm_strains,shear_strains=my_shear_strains)  #To modify the strains applied. 
        self.report(f'Deformations completed')    
        primitive = run_seekpath(self.ctx.ms_wc.outputs.output_structure)  #Check if primitive is different than conventional
        self.ctx.initial_vol = self.ctx.ms_wc.outputs.output_structure.get_cell_volume()
        self.ctx.primitive_vol = primitive.get_cell_volume()

    def geoopt_kpoints(self):
        if self.ctx.primitive_vol < self.ctx.initial_vol:
            #Setup for kpoint cellopt
            primitive = run_seekpath(self.ctx.ms_wc.outputs.output_structure)
            def_geokpoints_input = AttributeDict(self.exposed_inputs(Cp2kBaseWorkChain, namespace= 'cp2k_def_geo'))
            def_geokpoints_input['metadata'].update({'label': f'geo_kpoints','call_link_label': f'geo_kpoints'})
            def_geokpoints_input['cp2k']['structure'] = primitive

            yamlfullpath = self.ctx.protocol_geo
            with open(yamlfullpath, 'r') as stream:
                self.ctx.kpointsgeo_protocol = yaml.load(stream)
            self.ctx.kpointsgeo_settings = deepcopy(self.ctx.kpointsgeo_protocol['settings'])
            kinds = get_kinds_section(self.ctx.structure, self.ctx.kpointsgeo_protocol)
            merge_dict(self.ctx.kpointsgeo_settings, kinds)
            

            min_cell_size = Float(12)  # Min
            resize = check_resize_unit_cell_legacy(primitive, min_cell_size)  # Dict
            if resize['nx'] > 1 or resize['ny'] > 1 or resize['nz'] > 1:
                scheme = 'MONKHORST-PACK'
                scheme_grid = scheme + ' ' + str(resize['nx']) + ' ' + str(resize['ny']) + ' ' + str(resize['nz'])
                kpoints_dict = {
                    'FORCE_EVAL': {
                        'DFT': {
                            'KPOINTS': {
                                'SCHEME': scheme_grid,
                                'WAVEFUNCTIONS': 'COMPLEX',
                                'SYMMETRY': 'F',
                                'FULL_GRID': 'T',
                                'PARALLEL_GROUP_SIZE': 0
                            }
                        }
                    }
                }
                print('Using SCHEME MONKHORST-PACK with GRID {}x{}x{}'.format(resize['nx'], resize['ny'], resize['nz']))
            else:
                kpoints_dict = {
                        'FORCE_EVAL': {
                            'DFT': {
                                'KPOINTS': {
                                    'SCHEME': 'GAMMA',
                                    'WAVEFUNCTIONS': 'COMPLEX',
                                    'SYMMETRY': 'F',
                                    'FULL_GRID': 'T',
                                    'PARALLEL_GROUP_SIZE': 0
                                }
                            }
                        }
                    }
                print('At gamma point')

            merge_dict(self.ctx.kpointsgeo_settings, kpoints_dict)

            added_mos = {
                'FORCE_EVAL': {
                    'DFT': {
                        'SCF': {
                            'ADDED_MOS': add_condband(primitive)
                        }
                    }
                }
            }
            merge_dict(self.ctx.kpointsgeo_settings, added_mos)

            #Added MOs missing!

            def_geokpoints_input['cp2k']['parameters'] = Dict(dict=self.ctx.kpointsgeo_settings)

            running_base = self.submit(Cp2kBaseWorkChain, **def_geokpoints_input) #Submit the geo_opt
            self.report(f'Submit deformation Kpoints GEOOPT')
            self.to_context(geo_kpoints=running_base)


    def band_structure(self):
        #Band structure to compute effective mass 
        bands_cell_input = AttributeDict(self.exposed_inputs(Cp2kBandsWorkChain, namespace='cp2k_def_bands'))
        if self.ctx.primitive_vol < self.ctx.initial_vol:
            bands_cell_input['structure'] = structure_with_pbc(self.ctx.geo_kpoints.outputs.output_structure)
        else:
            bands_cell_input['structure'] = structure_with_pbc(self.ctx.ms_wc.outputs.output_structure)
        bands_cell_input['protocol_tag'] = Str(self.ctx.protocol_bands)
        bands_cell_input['metadata'].update({'label': f'bandstructure','call_link_label': f'bandstructure'})
        #bands_cell_input['protocol_tag'] = Str(os.path.join(THIS_DIR, 'mobility_protocols/bands_2'))
        running= self.submit(Cp2kBandsWorkChain, **bands_cell_input) #Submit the geo_opt 
        return ToContext(bands_wc=running)


    def run_geoopt_deformations(self):
        #Use ms_wc setting, and just modify for a restart 
        my_params_def_geo_modify = Dict(
            dict={
                'GLOBAL': {  
                    'RUN_TYPE': 'GEO_OPT'
                },
                'FORCE_EVAL': {
                    'STRESS_TENSOR': 'ANALYTICAL',
                    'PRINT': {
                        'STRESS_TENSOR': {
                            'ADD_LAST': 'SYMBOLIC',
                            'FILENAME': 'tensor',
                            'EACH':{
                                'CELL_OPT': '0',
                                'GEO_OPT': '1'
                            }
                        }
                    },

                
                    'DFT': {
                        'WFN_RESTART_FILE_NAME': './parent_calc/aiida-RESTART.wfn',
                        'SCF': {
                            'SCF_GUESS':
                                'RESTART',  # Use multistage wafefunction
                        },
                    }
                }
            }).store()   

        #Run all def GEOOPT
        for index,value in enumerate(self.ctx.def_dict):
            
            def_geo_input = AttributeDict(self.exposed_inputs(Cp2kBaseWorkChain, namespace= 'cp2k_def_geo'))
            def_geo_input['metadata'].update({'label': f'def_{index}','call_link_label': f'run_def_{index}'})
            def_geo_input['cp2k']['structure'] = StructureData(pymatgen_structure=value) 
            def_geo_input['cp2k']['parameters'] = aiida_dict_merge(self.ctx.ms_wc.outputs.last_input_parameters,my_params_def_geo_modify)
            def_geo_input['cp2k']['parent_calc_folder'] = self.ctx.ms_wc.outputs.remote_folder
            def_geo_input['cp2k']['settings'] = Dict(dict={'additional_retrieve_list': ['*.stress_tensor']})
            running_base = self.submit(Cp2kBaseWorkChain, **def_geo_input) #Submit the geo_opt
            self.report(f'Submit deformation {index}')
            self.to_context(**{f'def_geo_{index}': running_base})
    
    def run_bandsgeo_def(self):
        

        
        #Check if primitive is different than conventional
        #primitive = run_seekpath(self.ctx.ms_wc.outputs.output_structure)
        #self.ctx.initial_vol = self.ctx.ms_wc.outputs.output_structure.get_cell_volume()
        #self.ctx.primitive_vol = primitive.get_cell_volume()

        if self.ctx.primitive_vol < self.ctx.initial_vol:
            my_params_def_geo_modify = Dict(
            dict={
                'GLOBAL': {  
                    'RUN_TYPE': 'GEO_OPT'
                }
            }).store() 
            #primitive_struct = primitive.get_pymatgen_structure()
            primitive_struct  = self.ctx.geo_kpoints.outputs.output_structure.get_pymatgen_structure()
            my_norm_strains=[-0.02, -0.01, 0.01, 0.02]
            my_shear_strains=[-0.02,-0.01,0.01, 0.02]
            prim_defs = DeformedStructureSet(primitive_struct, norm_strains=my_norm_strains,shear_strains=my_shear_strains)
            
            my_strains_test = [Strain.from_deformation(Deformation(x)) for x in prim_defs.deformations]
            my_strains_voigt = [x.voigt for x in my_strains_test]
            self.ctx.voigt_round = np.around(my_strains_voigt, decimals=5)
            self.ctx.index_bands = []

            for index, my_norm in enumerate(self.ctx.voigt_round):
                if my_norm[0] != 0 or my_norm[1] != 0 or my_norm[2] != 0:
                    self.ctx.index_bands.append(index)
                    def_bgeo_input = AttributeDict(self.exposed_inputs(Cp2kBaseWorkChain, namespace= 'cp2k_def_geo'))
                    def_bgeo_input['metadata'].update({'label': f'def_B{index}','call_link_label': f'run_Bdef_{index}'})
                    def_bgeo_input['cp2k']['structure'] = StructureData(pymatgen_structure=prim_defs.deformed_structures[index]) 
                    def_bgeo_input['cp2k']['parameters'] = aiida_dict_merge(self.ctx.geo_kpoints.inputs.cp2k.parameters,my_params_def_geo_modify)                  
                    running_base = self.submit(Cp2kBaseWorkChain, **def_bgeo_input) #Submit the geo_opt
                    self.report(f'Submit deformation bgeo {index}')
                    self.to_context(**{f'def_bgeo_{index}': running_base})

    def run_bands_deformation(self):
        #Params
        if self.ctx.primitive_vol < self.ctx.initial_vol:

            for index, my_norm in enumerate(self.ctx.voigt_round):    
                if my_norm[0] != 0 or my_norm[1] != 0 or my_norm[2] != 0:
                    def_bands_input = AttributeDict(self.exposed_inputs(Cp2kBandsWorkChain, namespace='cp2k_def_bands'))
                    cp2k_def_geo_step = self.ctx[f'def_bgeo_{index}'].called[-1] 
                    def_bands_input['structure'] = structure_with_pbc(cp2k_def_geo_step.outputs.output_structure)
                    def_bands_input['metadata'].update({'label': f'def_banddef_{index}','call_link_label': f'run_banddef_{index}'})
                    def_bands_input['protocol_tag'] = Str(self.ctx.protocol_bands) 
                    running_bands = self.submit(Cp2kBandsWorkChain, **def_bands_input) #Submit the geo_opt 
                    self.to_context(**{f'def_bands_{index}': running_bands})


        else:   

            #Band structure of the deformed structures
            my_strains_test = [Strain.from_deformation(Deformation(x)) for x in self.ctx.def_dict.deformations]
            my_strains_voigt = [x.voigt for x in my_strains_test]
            self.ctx.voigt_round = np.around(my_strains_voigt, decimals=5)
            self.ctx.index_bands = []

            for index, my_norm in enumerate(self.ctx.voigt_round):
                
                if my_norm[0] != 0 or my_norm[1] != 0 or my_norm[2] != 0:
                    self.ctx.index_bands.append(index)
                    def_bands_input = AttributeDict(self.exposed_inputs(Cp2kBandsWorkChain, namespace='cp2k_def_bands'))
                    cp2k_def_geo_step = self.ctx[f'def_geo_{index}'].called[-1] 
                    def_bands_input['structure'] = structure_with_pbc(cp2k_def_geo_step.outputs.output_structure)
                    def_bands_input['metadata'].update({'label': f'def_banddef_{index}','call_link_label': f'run_banddef_{index}'})
                    def_bands_input['protocol_tag'] = Str(self.ctx.protocol_bands) 
                    running_bands = self.submit(Cp2kBandsWorkChain, **def_bands_input) #Submit the geo_opt 
                    self.to_context(**{f'def_bands_{index}': running_bands}) 

    def vacuum_pore(self):
        for index in self.ctx.index_bands:
            band_cp2k = self.ctx[f'def_bands_{index}'].called[0]
            band_calcjob = band_cp2k.called[0]
            input_vacuum = RemoteData(remote_path=band_calcjob.get_remote_workdir(), computer = band_calcjob.computer)
            VacCalc = CalculationFactory('lsmo.vac_vaccum')
            builder = VacCalc.get_builder()
            builder.folder = input_vacuum
            builder.metadata.options.max_wallclock_seconds = 1*3*60 #We can remove this line , test
            builder.code = Code.get_from_string('condapython3@daint2')
            running_vaccum_pore = self.submit(builder)
            self.to_context(**{f'def_band_{index}_pore': running_vaccum_pore})

        


    def results(self):
        #Collect all the results in a dictioary 

        results_dict = {}

        #
        # Effective mass
        #

        my_cp2k_calc_MS = self.ctx.ms_wc.called[1]  #Get the Cp2kBase from Multistage 
        my_cp2k_calc = my_cp2k_calc_MS.called[-1] # Get the Cp2kCalc from Cp2kBaseWC
        #my_filepath = last_stressfilepath(my_cp2k_calc.get_retrieved_node()._repository._get_base_folder().abspath)     


        #Hartree from bands 
        my_ws_bands = self.ctx.bands_wc.called[0]
        my_base_bands = my_ws_bands.called[-1]
        #my_bands_cube = my_base_bands.get_retrieved_node()._repository._get_base_folder().abspath + '/aiida-v_hartree-1_0.cube'

        my_opt_bands_info = get_info_fromBands(my_cp2k_calc.get_retrieved_node()._repository._get_base_folder().abspath +'/aiida.out')
        ms_bandgap_mid = self.ctx.ms_wc.outputs.output_parameters['final_bandgap_spin1_au'] * (HARTREE2EV/2)        
        my_opt_bands_pgm = get_pmg_bands(self.ctx.bands_wc.outputs.output_bands, self.ctx.bands_wc.inputs.structure, my_opt_bands_info['fermi']+ ms_bandgap_mid )
       


        my_vbm_info = my_opt_bands_pgm.get_vbm()
        my_cbm_info = my_opt_bands_pgm.get_cbm()
        my_cbm_info.pop('kpoint')
        my_vbm_info.pop('kpoint')
       
        results_dict['e_mass'] = call_sumo(my_opt_bands_pgm, 'electron')
        results_dict['h_mass'] = call_sumo(my_opt_bands_pgm, 'hole')
        results_dict['average_mass'] = (results_dict['e_mass'] + results_dict['h_mass'])/2
        results_dict['cbm'] = my_cbm_info['energy']
        results_dict['vbm'] = my_vbm_info['energy']

        #Defining variables
        energies_list = []   #List of energies 
        deformation_list = [] #List of deformation matrix
        deformation_list.append([[1,0,0],[0,1,0],[0,0,1]]) #identify in pos 0 to represent undeformed
        stress_tensor_list = [] #List of retrieved stress_tensor Units GPa

        #
        # Energy & Deformation matrices
        #
       
        energies_list.append(self.ctx.ms_wc.outputs.output_parameters['step_info']['energy_au'][-1] * HARTREE2EV )  #0 position is the one undeformed
        
        for index,value in enumerate(self.ctx.def_dict):
            energies_list.append(self.ctx[f'def_geo_{index}'].outputs.output_parameters['energy'] * HARTREE2EV)
            deformation_list.append(self.ctx.def_dict.deformations[index])

        self.report(f'Energies & Def MatricesDone!')

        #
        # Stress tensors 
        #

        #initial tensor, #Appending tensor 0
        my_cp2k_calc_MS = self.ctx.ms_wc.called[1]  #Get the Cp2kBase from Multistage 
        my_cp2k_calc = my_cp2k_calc_MS.called[-1] # Get the Cp2kCalc from Cp2kBaseWC
        temp_path = my_cp2k_calc.get_retrieved_node()._repository._get_base_folder().abspath + "/aiida-tensor-1_l.stress_tensor"
        temp_file = os.path.isfile(temp_path) 
        if temp_file == False:
            my_filepath = last_stressfilepath(my_cp2k_calc.get_retrieved_node()._repository._get_base_folder().abspath)
        else:
            my_filepath = temp_path
            self.report(f'Check convergence tensor 0')        

        with open(my_filepath, 'r') as my_tensor_matrix:
            my_data = my_tensor_matrix.readlines()[3:6]
            analytic = [ list(map(float, line.split()[2:])) for line in my_data]
            stress_tensor_list.append(analytic)
        
        #Appending the deformed stress tensors
        for index,value in enumerate(self.ctx.def_dict):
            cp2k_calc = self.ctx[f'def_geo_{index}'].called[-1]  # Get the Cp2kCalc from Cp2kBase
            temp_path =  cp2k_calc.get_retrieved_node()._repository._get_base_folder().abspath + "/aiida-tensor-1_l.stress_tensor"
            temp_file = os.path.isfile(temp_path)
            if temp_file == False:
                filepath =  last_stressfilepath(cp2k_calc.get_retrieved_node()._repository._get_base_folder().abspath)  
            else:
                filepath = temp_path
                self.report(f'Check convergence def {index}')
                
            with open(filepath, 'r') as my_tensor_matrix:
                my_data = my_tensor_matrix.readlines()[3:6]
                analytic = [ list(map(float, line.split()[2:])) for line in my_data]
                stress_tensor_list.append(analytic)

        self.report(f'Tensors Done!')  


        #
        # Deformation potential 
        #

        deformation_pot_cbm = []
        deformation_pot_vbm = []
        #for testing, to delete
        deformation_pot_cbm_min = []
        deformation_pot_vbm_min = []
        index_cbm = my_cbm_info['band_index'][Spin.up][0]
        kpoint_cbm = my_cbm_info['kpoint_index'][0]
        index_vbm = my_vbm_info['band_index'][Spin.up][0]
        kpoint_vbm = my_vbm_info['kpoint_index'][0]
        
        #pore potential
        center_pore =[]

        for index in self.ctx.index_bands:
            
            self.report(f'Def bands {index}')
            
            #if self.ctx.primitive_vol < self.ctx.initial_vol:
            #    temp_ws_bands = self.ctx[f'def_bgeo_{index}'].called[-1]
            #else:
            #    temp_ws_bands = self.ctx[f'def_geo_{index}'].called[-1]
            temp_ws_bands = self.ctx[f'def_geo_{index}'].called[-1]
            temp_bands_path = temp_ws_bands.get_retrieved_node()._repository._get_base_folder().abspath + '/aiida.out'
            temp_bands_info = get_info_fromBands(temp_bands_path)
            temp_bands_pgm = get_pmg_bands(self.ctx[f'def_bands_{index}'].outputs.output_bands, self.ctx[f'def_bands_{index}'].inputs.structure, temp_bands_info['fermi'] + ms_bandgap_mid )
   
            deformation_pot_cbm.append(float(temp_bands_pgm.bands[Spin.up][index_cbm][kpoint_cbm])-my_cbm_info['energy']) #If this is incorrect lets use the difference at cbm 
            deformation_pot_vbm.append(float(temp_bands_pgm.bands[Spin.up][index_vbm][kpoint_vbm])-my_vbm_info['energy'])
            temp_cbm = temp_bands_pgm.get_cbm()
            temp_vbm = temp_bands_pgm.get_vbm()
            self.report(temp_cbm['energy']) 
            self.report(temp_vbm['energy'])
            if temp_cbm['energy'] is not None and temp_vbm['energy'] is not None: 
                deformation_pot_cbm_min.append(temp_cbm['energy'] - my_cbm_info['energy']) #If this is incorrect lets use the difference at cbm 
                deformation_pot_vbm_min.append(temp_vbm['energy'] - my_vbm_info['energy'])
            else:
                deformation_pot_cbm_min.append("Issue")
                deformation_pot_cbm_min.append("Issue")

            #Hartree from bands 
            my_vacuum_bands = self.ctx[f'def_band_{index}_pore'].outputs.vac_level
            #my_base_bands = my_ws_bands.called[-1]
            #my_bands_cube = my_base_bands.get_retrieved_node()._repository._get_base_folder().abspath + '/aiida-v_hartree-1_0.cube' 
            #elec_pot = MOFVacLevel(my_bands_cube)
            #center_pore.append(elec_pot.get_vacuum_potential(threshold = 1.8, res=0.4, cube_size= [25, 25, 25]))
            center_pore.append(my_vacuum_bands.value)
        
        Def_fit_cbm = []
        Def_fit_vbm = []
        elec_pot_fit = []
        
        for i in range (0, len(self.ctx.index_bands), 4):
            my_mu = [sum(x) for x in self.ctx.voigt_round[i:i+4]]
            my_regression_cbm = linregress(my_mu,deformation_pot_cbm[i:i+4])
            my_regression_vbm = linregress(my_mu,deformation_pot_vbm[i:i+4])
            my_regression_elec = linregress(my_mu,center_pore[i:i+4])
            Def_fit_cbm.append(float(-1*my_regression_cbm.slope))
            Def_fit_vbm.append(float(-1*my_regression_vbm.slope))
            elec_pot_fit.append(float(-1*my_regression_elec.slope)) 
      



       

        # key in the dictionary with results ['energies','deformations','tensors']
        results_dict['energies'] = energies_list
        results_dict['deformations'] = deformation_list
        results_dict['tensors'] = stress_tensor_list
        results_dict['defpot_cbm'] = deformation_pot_cbm
        results_dict['defpot_vbm'] = deformation_pot_vbm
        results_dict['defpot_cbm_min'] = deformation_pot_cbm_min
        results_dict['defpot_vbm_min'] = deformation_pot_vbm_min
        results_dict['Def_potential_e'] = sum(Def_fit_cbm)/len(Def_fit_cbm)
        results_dict['Def_potential_h'] = sum(Def_fit_vbm)/len(Def_fit_vbm)
        #Tests
        results_dict['elec_pot_center'] = elec_pot_fit
        results_dict['Def_potential_e_elec'] = sum(np.array(Def_fit_cbm)-np.array(elec_pot_fit))/len(Def_fit_cbm)
        results_dict['Def_potential_h_elec'] = sum(np.array(Def_fit_vbm)-np.array(elec_pot_fit))/len(Def_fit_vbm)
        

        #Bulk Modulus
        my_stresses = [Stress(x) for x in results_dict['tensors']]
        my_strains = [Strain.from_deformation(Deformation(x)) for x in results_dict['deformations']]
        my_fit_tensor = ElasticTensor.from_independent_strains(my_strains[1:],my_stresses[1:])
        results_dict['k_voigt'] = (-1)*my_fit_tensor.k_voigt 
        #ElasticTensor.from_independent_strains(my_strains[1:],my_stresses[1:]) # lets use this one instead ! 
        #Equation k_voigt= Bulk Modulus in GPA , def_pot = Def potential in eV , avg_mas = effective mass in me units
        results_dict['mu_BS_e'] = (10000 * CONSTANT * QCHARGE_C * (results_dict['k_voigt']) * GPA_TO_PA * math.pow(KB_T_J,-1.5))/(math.pow((results_dict['Def_potential_e'] * EV2J),2)*  math.pow((results_dict['e_mass'] * MASSELECTRON_KG),2.5))
        results_dict['mu_BS_h'] = (10000 * CONSTANT * QCHARGE_C * (results_dict['k_voigt']) * GPA_TO_PA * math.pow(KB_T_J,-1.5))/(math.pow((results_dict['Def_potential_h'] * EV2J),2)*  math.pow((results_dict['h_mass'] * MASSELECTRON_KG),2.5))
      
        results_dict['mu_BS_e_elec'] = (10000 * CONSTANT * QCHARGE_C * (results_dict['k_voigt']) * GPA_TO_PA * math.pow(KB_T_J,-1.5))/(math.pow((results_dict['Def_potential_e_elec'] * EV2J),2)*  math.pow((results_dict['e_mass'] * MASSELECTRON_KG),2.5))
        results_dict['mu_BS_h_elec'] = (10000 * CONSTANT * QCHARGE_C * (results_dict['k_voigt']) * GPA_TO_PA * math.pow(KB_T_J,-1.5))/(math.pow((results_dict['Def_potential_h_elec'] * EV2J),2)*  math.pow((results_dict['h_mass'] * MASSELECTRON_KG),2.5))
      


        aiida_result_dict = Dict(dict=results_dict) 
        aiida_result_dict.store()

        self.out('out_data', aiida_result_dict)
        self.report(f'BS MobilityWorkChain Completed')

        







     





        


        

        


